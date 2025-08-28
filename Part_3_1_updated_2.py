# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:59:31 2025

@author: nfpm5
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:05:01 2025

@author: nfpm5
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# === Load Dataset ===
data_path = "data_for_energyy.csv"
df = pd.read_csv(data_path, parse_dates=['time'])
df.set_index('time', inplace=True)
cols_required = ['pv_production', 'wind_production', 'consumption', 'spot_market_price']
df = df[[col for col in cols_required if col in df.columns]].copy()
df[df < 0] = 0  # Enforce physical constraint: no negative values
df['gen'] = df['pv_production'] + df['wind_production']

# === Scenario Construction ===
scenarios = {}

# Scenario 1: Extreme Renewable Drought
scen1 = df.copy()
scen1['pv_production'] *= 0.05
scen1['wind_production'] *= 0.05
scen1['gen'] = scen1['pv_production'] + scen1['wind_production']
scenarios['extreme_renewable_drought'] = scen1

# Scenario 2: Polar Vortex (price + demand spike)
scen2 = df.copy()
scen2['consumption'] *= 1.6
scen2['spot_market_price'] *= 1.5
scenarios['polar_vortex'] = scen2

# Scenario 3: Evening Peak Load
scen3 = df.copy()
evening_hours = scen3.index.hour.isin([17, 18, 19, 20])
scen3.loc[evening_hours, 'consumption'] *= 1.8
scenarios['evening_peak'] = scen3

# Scenario 4: Spring Ramp Up
scen4 = df.copy()
spring_hours = scen4.index.month.isin([3, 4, 5])
spring_df = scen4[spring_hours].copy()
spring_df['pv_production'] *= 1.4
spring_df['wind_production'] *= 0.8
spring_df['consumption'] *= 0.9
spring_df['gen'] = spring_df['pv_production'] + spring_df['wind_production']
scenarios['spring_ramp'] = spring_df

# Scenario 5: Weekend Dip
scen5 = df.copy()
weekend_days = scen5.index.dayofweek.isin([5, 6])
scen5.loc[weekend_days, 'consumption'] *= 0.7
scen5['gen'] = scen5['pv_production'] + scen5['wind_production']
scenarios['weekend_dip'] = scen5

# Scenario 6: Battery Stress Test
scen6 = df.copy()
np.random.seed(42)
fluctuations = 1 + 0.2 * np.sin(np.linspace(0, 20 * np.pi, len(scen6)))
scen6['consumption'] *= fluctuations
scenarios['battery_stress_test'] = scen6

# === Constants ===
BATTERY_CAP = 100.0     # kWh
HYDROGEN_CAP = 200.0    # kWh
SOC_MIN, SOC_MAX = 15.0, 90.0
EFF_BATTERY_CH, EFF_BATTERY_DIS = 0.95, 0.95
EFF_H2_EL, EFF_H2_FC = 0.70, 0.55
INIT_BATT, INIT_H2 = 50.0, 100.0

# === Custom Loss Function (Energy-Economic Weighted)
def compute_loss(unmet, curtail, grid_import, price):
    alpha = 1.0  # weight on unmet load penalty
    beta = 0.2   # weight on curtailment penalty
    gamma = 0.5  # weight on grid dependency cost

    loss = (
        alpha * np.square(unmet).sum() +
        beta * np.square(curtail).sum() +
        gamma * np.sum(grid_import * price)
    )
    return loss

# === Dispatch Strategy (CADT-PIML vs Rule-Based)
def dispatch_energy(soc_batt_percent, soc_h2_percent, gen, demand, price, cidt=False):
    battery_soc_trace, h2_soc_trace = [], []
    unmet_trace, cost_trace, curtail_trace = [], [], []

    batt_kwh = (soc_batt_percent / 100) * BATTERY_CAP
    h2_kwh = (soc_h2_percent / 100) * HYDROGEN_CAP

    batt_min_kwh = SOC_MIN / 100 * BATTERY_CAP
    batt_max_kwh = SOC_MAX / 100 * BATTERY_CAP
    h2_min_kwh = SOC_MIN / 100 * HYDROGEN_CAP
    h2_max_kwh = SOC_MAX / 100 * HYDROGEN_CAP

    for t in range(len(demand)):
        net_load = demand[t] - gen[t]
        p_now = price[t]
        p_next = price[t + 1] if t + 1 < len(price) else p_now
        price_rise = p_next > p_now * 1.01
        price_drop = p_now > p_next * 1.01

        unmet = curtail = grid_draw = 0

        if net_load > 0:
            # Discharging
            max_batt_dis = max(0, batt_kwh - batt_min_kwh)
            batt_factor = 0.6 if cidt and price_rise else 1.0
            batt_dis = min(net_load / EFF_BATTERY_DIS, max_batt_dis * batt_factor)
            batt_dis = min(batt_dis, batt_kwh - batt_min_kwh)
            batt_kwh -= batt_dis
            net_load -= batt_dis * EFF_BATTERY_DIS

            max_h2_dis = max(0, h2_kwh - h2_min_kwh)
            h2_factor = 1.2 if cidt and price_rise else 1.0
            h2_dis = min(net_load / EFF_H2_FC, max_h2_dis * h2_factor)
            h2_dis = min(h2_dis, h2_kwh - h2_min_kwh)
            h2_kwh -= h2_dis
            net_load -= h2_dis * EFF_H2_FC

            if net_load > 0:
                grid_draw = net_load

        else:
            # Charging
            surplus = -net_load

            max_batt_ch = max(0, batt_max_kwh - batt_kwh)
            batt_factor = 1.2 if cidt and price_drop else 1.0
            batt_ch = min(surplus * EFF_BATTERY_CH, max_batt_ch * batt_factor)
            batt_ch = min(batt_ch, batt_max_kwh - batt_kwh)
            batt_kwh += batt_ch
            surplus -= batt_ch / EFF_BATTERY_CH

            max_h2_ch = max(0, h2_max_kwh - h2_kwh)
            h2_factor = 1.1 if cidt else 1.0
            h2_ch = min(surplus / EFF_H2_EL, max_h2_ch * h2_factor)
            h2_ch = min(h2_ch, h2_max_kwh - h2_kwh)
            h2_kwh += h2_ch
            surplus -= h2_ch * EFF_H2_EL

            if surplus > 0:
                curtail = surplus

        # Final safety clamps
        batt_kwh = np.clip(batt_kwh, batt_min_kwh, batt_max_kwh)
        h2_kwh = np.clip(h2_kwh, h2_min_kwh, h2_max_kwh)

        battery_soc_trace.append((batt_kwh / BATTERY_CAP) * 100)
        h2_soc_trace.append((h2_kwh / HYDROGEN_CAP) * 100)
        unmet_trace.append(grid_draw)
        cost_trace.append(grid_draw * p_now)
        curtail_trace.append(curtail)

    return {
        'battery_soc': battery_soc_trace,
        'hydrogen_soc': h2_soc_trace,
        'unmet_load': unmet_trace,
        'grid_cost': cost_trace,
        'curtailment': curtail_trace,
        'loss_value': compute_loss(unmet_trace, curtail_trace, unmet_trace, price)
    }

# === Run Simulations for All Scenarios ===
warem_results = {}

for scen_name, scen_df in scenarios.items():
    gen = scen_df['gen'].values
    load = scen_df['consumption'].values
    price = scen_df['spot_market_price'].values

    # Rule-Based (no CADT-PIML)
    result_rule = dispatch_energy(
        soc_batt_percent=INIT_BATT,
        soc_h2_percent=INIT_H2,
        gen=gen,
        demand=load,
        price=price,
        cidt=False
    )

    # CADT-PIML-Controlled Strategy
    result_cidt = dispatch_energy(
        soc_batt_percent=INIT_BATT,
        soc_h2_percent=INIT_H2,
        gen=gen,
        demand=load,
        price=price,
        cidt=True
    )

    warem_results[scen_name] = {
        'Rule-Based': result_rule,
        'CADT-PIML': result_cidt
    }

# === Metric Extraction ===
def extract_metrics(results):
    return {
        'Unmet Load (kWh)': round(np.sum(results['unmet_load']), 2),
        'Grid Cost (£)': round(np.sum(results['grid_cost']), 2),
        'Curtailment (kWh)': round(np.sum(results['curtailment']), 2),
        'Avg Battery SoC (%)': round(np.mean(results['battery_soc']), 2),
        'Avg Hydrogen SoC (%)': round(np.mean(results['hydrogen_soc']), 2),
        'Total Loss Value': round(results['loss_value'], 2)
    }

metrics_all = []
for scen_name, strat_results in warem_results.items():
    for strategy, result in strat_results.items():
        entry = extract_metrics(result)
        entry['Scenario'] = scen_name
        entry['Strategy'] = strategy
        metrics_all.append(entry)

df_metrics = pd.DataFrame(metrics_all)
df_metrics = df_metrics[[
    'Scenario', 'Strategy', 'Unmet Load (kWh)', 'Grid Cost (£)',
    'Curtailment (kWh)', 'Avg Battery SoC (%)', 'Avg Hydrogen SoC (%)',
    'Total Loss Value'
]]

# Save Results
results_dir = "Results"
os.makedirs(results_dir, exist_ok=True)
df_metrics.to_csv(os.path.join(results_dir, "WAREM_Metrics_Loss_Integrated.csv"), index=False)

# === Visualisation of State of Charge and Loss ===
for scen_name, strat_data in warem_results.items():
    time = range(len(next(iter(strat_data.values()))['battery_soc']))

    # Plot Battery SoC
    plt.figure(figsize=(12, 5))
    for label, result in strat_data.items():
        plt.plot(time, result['battery_soc'], label=f"{label} – Battery")
    plt.axhline(SOC_MIN, linestyle='--', color='gray', label='Min SoC')
    plt.axhline(SOC_MAX, linestyle='--', color='black', label='Max SoC')
    plt.title(f"Battery SoC – {scen_name.replace('_', ' ').title()}")
    plt.xlabel("Time [h]")
    plt.ylabel("State of Charge [%]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"SoC_Battery_{scen_name}.png"))
    plt.close()

    # Plot Hydrogen SoC
    plt.figure(figsize=(12, 5))
    for label, result in strat_data.items():
        plt.plot(time, result['hydrogen_soc'], label=f"{label} – HSL")
    plt.axhline(SOC_MIN, linestyle='--', color='gray', label='Min HSL')
    plt.axhline(SOC_MAX, linestyle='--', color='black', label='Max HSL')
    plt.title(f"HSL – {scen_name.replace('_', ' ').title()}")
    plt.xlabel("Time [h]")
    plt.ylabel("HSL [%]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"HSL_Hydrogen_{scen_name}.png"))
    plt.close()

# === Plot Restoration Curves ===
for scen_name, strat_data in warem_results.items():
    time = range(len(next(iter(strat_data.values()))['unmet_load']))
    plt.figure(figsize=(12, 5))
    for label, result in strat_data.items():
        plt.plot(time, np.cumsum(result['unmet_load']), label=label)
    plt.title(f"Load Restoration Curve – {scen_name.replace('_', ' ').title()}")
    plt.xlabel("Time [h]")
    plt.ylabel("Cumulative Unmet Load [kWh]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"Restoration_{scen_name}.png"))
    plt.close()






fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
axs = axs.flatten()

for i, (scen_name, strat_data) in enumerate(warem_results.items()):
    time = range(len(next(iter(strat_data.values()))['unmet_load']))
    ax = axs[i]
    for label, result in strat_data.items():
        ax.plot(time, np.cumsum(result['unmet_load']), label=label)
    ax.set_title(scen_name.replace('_', ' ').title())
    ax.set_ylabel("Cumulative Unmet Load [kWh]")
    ax.grid(True)
    if i >= 4:
        ax.set_xlabel("Time [h]")
    if i == 0:
        ax.legend()

plt.suptitle("Load Restoration Curves for All Scenarios", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Box_Restoration_Curves.png"), dpi=600)
plt.close()

fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
axs = axs.flatten()

for i, (scen_name, strat_data) in enumerate(warem_results.items()):
    time = range(len(next(iter(strat_data.values()))['battery_soc']))
    ax = axs[i]
    for label, result in strat_data.items():
        ax.plot(time, result['battery_soc'], label=label)
    ax.axhline(SOC_MIN, linestyle='--', color='gray', linewidth=1)
    ax.axhline(SOC_MAX, linestyle='--', color='black', linewidth=1)
    ax.set_title(scen_name.replace('_', ' ').title())
    ax.set_ylabel("Battery SoC [%]")
    ax.grid(True)
    if i >= 4:
        ax.set_xlabel("Time [h]")
    if i == 0:
        ax.legend()

plt.suptitle("Battery State of Charge (SoC) for All Scenarios", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Box_Battery_SOC_AllScenarios.png"), dpi=600)
plt.close()

fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
axs = axs.flatten()

for i, (scen_name, strat_data) in enumerate(warem_results.items()):
    time = range(len(next(iter(strat_data.values()))['hydrogen_soc']))
    ax = axs[i]
    for label, result in strat_data.items():
        ax.plot(time, result['hydrogen_soc'], label=label)
    ax.axhline(SOC_MIN, linestyle='--', color='gray', linewidth=1)
    ax.axhline(SOC_MAX, linestyle='--', color='black', linewidth=1)
    ax.set_title(scen_name.replace('_', ' ').title())
    ax.set_ylabel("HSL [%]")
    ax.grid(True)
    if i >= 4:
        ax.set_xlabel("Time [h]")
    if i == 0:
        ax.legend()

plt.suptitle("HSL for All Scenarios", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Box_Hydrogen_HSL_AllScenarios.png"), dpi=600)
plt.close()


import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=False)
axs = axs.flatten()

for i, (scen_name, scen_df) in enumerate(scenarios.items()):
    ax = axs[i]
    df_sub = scen_df.head(100)  # First 100 time steps
    time = df_sub.index

    ax.plot(time, df_sub['pv_production'], label='PV', color='gold', linewidth=1.5)
    ax.plot(time, df_sub['wind_production'], label='Wind', color='skyblue', linewidth=1.5)
    ax.plot(time, df_sub['consumption'], label='Load', color='salmon', linewidth=1.8)

    ax.set_title(scen_name.replace('_', ' ').title(), fontsize=16)
    ax.set_ylabel("Power [kW]", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=30)

    if i == 0:
        ax.legend(fontsize=14)
    if i >= 4:
        ax.set_xlabel("Time", fontsize=16)

plt.suptitle("Scenario Profiles – PV, Wind, and Load", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Scenario_Profiles_PV_Wind_Load_Corrected.png"), dpi=600)
plt.show()


import matplotlib.pyplot as plt

# === Plot Battery SoC (3x2 grid) ===
fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axs = axs.flatten()

for i, (scen_name, strat_data) in enumerate(warem_results.items()):
    ax = axs[i]
    time = range(len(next(iter(strat_data.values()))['battery_soc']))
    
    ax.plot(time, strat_data['Rule-Based']['battery_soc'], label='Rule-Based', linewidth=2)
    ax.plot(time, strat_data['CADT-PIML']['battery_soc'], label='CADT-PIML', linewidth=2, linestyle='--')
    
    ax.axhline(SOC_MIN, linestyle=':', color='red', linewidth=2.0)
    ax.axhline(SOC_MAX, linestyle=':', color='black', linewidth=2.0)
    
    ax.set_title(scen_name.replace('_', ' ').title(), fontsize=16)
    ax.set_ylabel("Battery SoC [%]", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True)

    if i >= 4:
        ax.set_xlabel("Time [h]", fontsize=16)
    if i == 0:
        ax.legend(fontsize=14)

plt.suptitle("Battery State of Charge (SoC) – All Scenarios", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Battery_SOC_All_Scenarios.png"), dpi=600)
plt.show()


# === Plot Hydrogen SoC (3x2 grid) ===
fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axs = axs.flatten()

for i, (scen_name, strat_data) in enumerate(warem_results.items()):
    ax = axs[i]
    time = range(len(next(iter(strat_data.values()))['hydrogen_soc']))
    
    ax.plot(time, strat_data['Rule-Based']['hydrogen_soc'], label='Rule-Based', linewidth=2)
    ax.plot(time, strat_data['CADT-PIML']['hydrogen_soc'], label='CADT-PIML', linewidth=2, linestyle='--')
    
    ax.axhline(SOC_MIN, linestyle=':', color='red', linewidth=2.0)
    ax.axhline(SOC_MAX, linestyle=':', color='black', linewidth=2.0)
    
    ax.set_title(scen_name.replace('_', ' ').title(), fontsize=16)
    ax.set_ylabel("HSL [%]", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True)

    if i >= 4:
        ax.set_xlabel("Time [h]", fontsize=16)
    if i == 0:
        ax.legend(fontsize=14)

plt.suptitle("HSL – All Scenarios", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Hydrogen_HSL_All_Scenarios.png"), dpi=600)
plt.show()












import matplotlib.pyplot as plt

# === Settings ===
cidt_color = '#2ca02c'  # Vibrant green for visibility
title_prefixes = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
title_bodies = [name.replace('_', ' ').title() for name in warem_results.keys()]

# === Plot Battery SoC (CADT-PIML only) ===
fig, axs = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
axs = axs.flatten()

for i, (scen_name, strat_data) in enumerate(warem_results.items()):
    ax = axs[i]
    time = range(len(strat_data['CADT-PIML']['battery_soc']))

    ax.plot(time, strat_data['CADT-PIML']['battery_soc'],
            label='CADT-PIML', linewidth=1.0, color=cidt_color)

    ax.axhline(SOC_MIN, linestyle='--', color='red', linewidth=2.0, label='Min SoC' if i == 0 else "")
    ax.axhline(SOC_MAX, linestyle='--', color='black', linewidth=2.0, label='Max SoC' if i == 0 else "")

    ax.set_title(f"$\\bf{{{title_prefixes[i]}}}$ {title_bodies[i]}", fontsize=16)
    ax.set_ylabel("Battery SoC [%]", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)

    if i >= 4:
        ax.set_xlabel("Time [h]", fontsize=16)
    if i == 0:
        ax.legend(fontsize=14)

plt.suptitle("Battery State of Charge (SoC) – CADT-PIML", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Battery_SOC_CADT_Only.png"), dpi=600)
plt.show()



import matplotlib.pyplot as plt

# Title formatting
title_prefixes = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
title_bodies = [name.replace('_', ' ').title() for name in warem_results.keys()]

# === Plot Hydrogen SoC (CADT only) ===
fig, axs = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
axs = axs.flatten()

for i, (scen_name, strat_data) in enumerate(warem_results.items()):
    ax = axs[i]
    time = range(len(strat_data['CADT-PIML']['hydrogen_soc']))

    ax.plot(time, strat_data['CADT-PIML']['hydrogen_soc'],
            label='CADT-PIML', linewidth=1.0, color=cidt_color)

    ax.axhline(SOC_MIN, linestyle='--', color='red', linewidth=2.0, label='Min HSL' if i == 0 else "")
    ax.axhline(SOC_MAX, linestyle='--', color='black', linewidth=2.0, label='Max HSL' if i == 0 else "")

    ax.set_title(f"$\\bf{{{title_prefixes[i]}}}$ {title_bodies[i]}", fontsize=16)
    ax.set_ylabel("HSL [%]", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)

    if i >= 4:
        ax.set_xlabel("Time [h]", fontsize=16)
    if i == 0:
        ax.legend(fontsize=14)

plt.suptitle("HSL – CADT-PIML", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Hydrogen_HSL_CIDT_Only.png"), dpi=600)
plt.show()






import matplotlib.pyplot as plt

# Title formatting
title_prefixes = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
title_bodies = [name.replace('_', ' ').title() for name in warem_results.keys()]

# === 1. Restoration Curve (CADT Only) ===
fig, axs = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
axs = axs.flatten()

for i, (scen_name, strat_data) in enumerate(warem_results.items()):
    ax = axs[i]
    cidt_unmet = strat_data['CADT-PIML']['unmet_load']
    time = range(len(cidt_unmet))

    ax.plot(time, np.cumsum(cidt_unmet), label='CADT-PIML', color='green', linewidth=2)
    ax.set_title(f"$\\bf{{{title_prefixes[i]}}}$ {title_bodies[i]}", fontsize=16)
    ax.set_ylabel("CUL [kWh]", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True)

    if i >= 4:
        ax.set_xlabel("Time [h]", fontsize=16)
    if i == 0:
        ax.legend(fontsize=14)

plt.suptitle("Load Restoration Curves – CADT-PIML", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Restoration_Curves_CADT_Only.png"), dpi=600)
plt.show()



import matplotlib.pyplot as plt

# Split bold tag and title body
title_prefixes = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
title_bodies = [
    "Extreme Renewable Drought",
    "Polar Vortex",
    "Evening Peak",
    "Spring Ramp",
    "Weekend Dip",
    "Battery Stress Test"
]

fig, axs = plt.subplots(3, 2, figsize=(18, 14), sharex=True)
axs = axs.flatten()

for i, (scen_name, scen_df) in enumerate(scenarios.items()):
    ax = axs[i]
    df_sub = scen_df.copy()
    time = range(len(df_sub))  # time in hours

    ax.plot(time, df_sub['pv_production'], label='PV', color='gold', linewidth=1)
    ax.plot(time, df_sub['wind_production'], label='Wind', color='skyblue', linewidth=1)
    ax.plot(time, df_sub['consumption'], label='Load', color='salmon', linewidth=1.2)

    # Title with bold prefix only
    ax.set_title(f"$\\bf{{{title_prefixes[i]}}}$ {title_bodies[i]}", fontsize=16)
    ax.set_ylabel("Power [kW]", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_xticklabels(['0', '2000', '4000', '6000', '8000'])

    if i == 0:
        ax.legend(
            fontsize=16,
            loc='upper center',
            ncol=3,
            frameon=True,
            bbox_to_anchor=(0.5, 1.0),
            borderpad=0.8,
            columnspacing=1.5
        )

    if i >= 4:
        ax.set_xlabel("Time [h]", fontsize=16)

plt.suptitle("Scenario Profiles (Full Year – 8760h) – PV, Wind, and Load", fontsize=18)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(os.path.join(results_dir, "Scenario_Profiles_FullYear_HourTicks.png"), dpi=600)
plt.show()


