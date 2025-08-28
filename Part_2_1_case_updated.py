# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:37:45 2025

@author: nfpm5
"""

# -*- coding: utf-8 -*-
"""
Improved CADT vs Rule-Based Strategy Simulation
Physics-Informed Version
Last Updated: June 2025
Author: cavus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Paths ===
data_path = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Good work_2\data_for_energyy.csv"
results_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Good work_2\Results"
os.makedirs(results_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(data_path, parse_dates=['time'])
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
df.set_index('time', inplace=True)

df = df[['pv_production', 'wind_production', 'consumption', 'spot_market_price']].copy()
df = df.sort_index()
df['future_price'] = df['spot_market_price'].shift(-1).fillna(method='ffill')
df['gen'] = df['pv_production'] + df['wind_production']

# === Constants ===
BATTERY_CAP = 120  # kWh
HYDROGEN_CAP = 75  # kWh
SOC_MIN = 15.0     # %
SOC_MAX = 90.0     # %
INIT_BATTERY = 70.0    # Initial battery SoC (%)
INIT_HYDROGEN = 70.0   # Initial H2 SoC (%)

# === Temperature-Aware Efficiency Models (Physics-Informed) ===
def thermal_eff_battery(T=25):
    alpha = 0.01
    return 0.93 * (1 + alpha * (T - 25))  # Decreases/increases with T

def thermal_eff_h2(T=25):
    gamma = 0.008
    return 0.80 * (1 + gamma * (T - 25))  # Electrolyser efficiency

# === SoC Update Function with Physical Constraints ===
def update_soc(soc, charge, discharge, eff_ch, eff_dis, capacity):
    soc_kwh = soc / 100 * capacity
    soc_kwh_new = soc_kwh + eff_ch * charge - discharge / eff_dis
    soc_kwh_new = np.clip(soc_kwh_new, SOC_MIN / 100 * capacity, SOC_MAX / 100 * capacity)
    soc_percent = (soc_kwh_new / capacity) * 100
    energy_in = eff_ch * charge
    energy_out = discharge / eff_dis
    return soc_percent, energy_in, energy_out

# === Strategy Simulation Function ===
def run_strategy(df, cidt=False, temp_c=25):
    soc_batt, soc_h2 = INIT_BATTERY, INIT_HYDROGEN
    soc_batt_list, soc_h2_list, grid_list = [], [], []
    batt_eff_in_list, batt_eff_out_list = [], []
    h2_eff_in_list, h2_eff_out_list = [], []
    physics_loss_list = []

    eff_batt_ch = thermal_eff_battery(temp_c)
    eff_batt_dis = eff_batt_ch
    eff_h2_el = thermal_eff_h2(temp_c)
    eff_h2_fc = 0.65 * (1 + 0.005 * (temp_c - 25))  # moderate temperature sensitivity

    for idx, row in df.iterrows():
        gen = max(0, row['gen'])
        demand = row['consumption']
        net = gen - demand
        p_now = row['spot_market_price']
        p_future = row['future_price']

        anticipatory_charge = cidt and (p_future > p_now * 1.005)
        anticipatory_discharge = cidt and (p_now > p_future * 1.005)

        batt_eff_in = batt_eff_out = h2_eff_in = h2_eff_out = 0.0

        net_original = net  # For loss tracking

        if net >= 0:
            batt_ch = min(net * 0.7, BATTERY_CAP * (SOC_MAX - soc_batt) / 100.0) / eff_batt_ch
            if cidt and anticipatory_charge:
                batt_ch *= 1.7
            soc_batt, batt_eff_in, _ = update_soc(soc_batt, batt_ch, 0, eff_batt_ch, eff_batt_dis, BATTERY_CAP)
            net -= batt_ch

            h2_ch = min(net, HYDROGEN_CAP * (SOC_MAX - soc_h2) / 100.0) / eff_h2_el
            if cidt and anticipatory_charge:
                h2_ch *= 1.3
            soc_h2, h2_eff_in, _ = update_soc(soc_h2, h2_ch, 0, eff_h2_el, eff_h2_fc, HYDROGEN_CAP)
            net -= h2_ch

            grid_import = 0

        else:
            batt_dis = min(-net * 0.6, BATTERY_CAP * (soc_batt - SOC_MIN) / 100.0) * eff_batt_dis
            if cidt and anticipatory_discharge:
                batt_dis *= 1.7
            soc_batt, _, batt_eff_out = update_soc(soc_batt, 0, batt_dis, eff_batt_ch, eff_batt_dis, BATTERY_CAP)
            net += batt_dis

            h2_dis = min(-net, HYDROGEN_CAP * (soc_h2 - SOC_MIN) / 100.0) * eff_h2_fc
            if cidt and anticipatory_discharge:
                h2_dis *= 1.4
            soc_h2, _, h2_eff_out = update_soc(soc_h2, 0, h2_dis, eff_h2_el, eff_h2_fc, HYDROGEN_CAP)
            net += h2_dis

            grid_import = max(0, -net)

        # === Physics Loss Proxy ===
        physics_net_balance = gen - (demand + grid_import) - batt_eff_in - h2_eff_in + batt_eff_out + h2_eff_out
        physics_loss = abs(physics_net_balance) / (gen + 1e-6)
        physics_loss_list.append(physics_loss)

        # === Store Results ===
        soc_batt_list.append(soc_batt)
        soc_h2_list.append(soc_h2)
        grid_list.append(grid_import)
        batt_eff_in_list.append(batt_eff_in)
        batt_eff_out_list.append(batt_eff_out)
        h2_eff_in_list.append(h2_eff_in)
        h2_eff_out_list.append(h2_eff_out)

    return (soc_batt_list, soc_h2_list, grid_list,
            batt_eff_in_list, batt_eff_out_list, h2_eff_in_list, h2_eff_out_list,
            physics_loss_list)

# === Run Simulations ===
results_rule = run_strategy(df, cidt=False)
results_cidt = run_strategy(df, cidt=True)

# === Unpack Results ===
(df['battery_soc_rule'], df['hydrogen_soc_rule'], df['grid_import_rule'],
 df['batt_in_rule'], df['batt_out_rule'],
 df['h2_in_rule'], df['h2_out_rule'],
 df['physics_loss_rule']) = results_rule

(df['battery_soc_cidt'], df['hydrogen_soc_cidt'], df['grid_import_cidt'],
 df['batt_in_cidt'], df['batt_out_cidt'],
 df['h2_in_cidt'], df['h2_out_cidt'],
 df['physics_loss_cidt']) = results_cidt

# === Compute Metrics ===
df['cost_rule'] = df['grid_import_rule'] * df['spot_market_price']
df['cost_cidt'] = df['grid_import_cidt'] * df['spot_market_price']
df['curtail_rule'] = (df['gen'] - df['consumption'] - df['grid_import_rule']).clip(lower=0)
df['curtail_cidt'] = (df['gen'] - df['consumption'] - df['grid_import_cidt']).clip(lower=0)

# === Save Summary ===
df_summary = pd.DataFrame({
    "Metric": [
        "Total Grid Import (kWh)",
        "Total Grid Cost (£)",
        "Total Curtailment (kWh)",
        "Avg Battery SoC (%)",
        "Avg Hydrogen SoC (%)",
        "Avg Physics Loss (Rule)",
        "Avg Physics Loss (CADT)"
    ],
    "Rule-Based": [
        df['grid_import_rule'].sum(),
        df['cost_rule'].sum(),
        df['curtail_rule'].sum(),
        df['battery_soc_rule'].mean(),
        df['hydrogen_soc_rule'].mean(),
        df['physics_loss_rule'].mean(),
        np.nan
    ],
    "CADT-Control": [
        df['grid_import_cidt'].sum(),
        df['cost_cidt'].sum(),
        df['curtail_cidt'].sum(),
        df['battery_soc_cidt'].mean(),
        df['hydrogen_soc_cidt'].mean(),
        np.nan,
        df['physics_loss_cidt'].mean()
    ]
})

summary_path = os.path.join(results_dir, "Performance_Summary_Comparison.csv")
df_summary.to_csv(summary_path, index=False)
print(f"✅ Summary saved to {summary_path}")

# === Time Series Plots ===
def save_timeseries_plot(y1, y2, ylabel, title, fname, index=None):
    plt.figure(figsize=(12, 5))
    x = index if index is not None else df.index
    plt.plot(x, y1, label='Rule-Based', linestyle='--', color='red')
    plt.plot(x, y2, label='CADT-PIML', color='seagreen')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, fname), dpi=600)
    plt.close()


save_timeseries_plot(df['cost_rule'], df['cost_cidt'], "Grid Import Cost (£)", "Grid Cost – Full Period", "Grid_Cost_TimeSeries.png")
save_timeseries_plot(df['curtail_rule'], df['curtail_cidt'], "Curtailment (kWh)", "Curtailment – Full Period", "Curtailment_TimeSeries.png")
save_timeseries_plot(df['physics_loss_rule'], df['physics_loss_cidt'], "Physics-Informed Loss", "Physical Consistency Deviation", "Physics_Loss_TimeSeries.png")


# === Overlay Last 100 Hours ===
df_last100 = df.tail(100)

def overlay_soc_plot(col_rule, col_cidt, ylabel, title, fname):
    plt.figure(figsize=(12, 5))
    plt.plot(df_last100.index, df_last100[col_rule], '--', label=f'{ylabel} (Rule-Based)', color='crimson')
    plt.plot(df_last100.index, df_last100[col_cidt], label=f'{ylabel} (CADT-PIML)', color='seagreen')
    plt.axhline(SOC_MIN, linestyle=':', color='red', label='Min SoC (15%)')
    plt.axhline(SOC_MAX, linestyle=':', color='darkred', label='Max SoC (90%)')
    plt.ylabel(f"{ylabel} [%]")
    plt.xlabel("Date & Time")
    plt.title(f"{title} – Last 100 Hours")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, fname), dpi=600)
    plt.close()

overlay_soc_plot('battery_soc_rule', 'battery_soc_cidt', "Battery SoC", "Battery SoC", "Battery_SOC_Last100h.png")
overlay_soc_plot('hydrogen_soc_rule', 'hydrogen_soc_cidt', "HSL", "HSL", "Hydrogen_SOC_Last100h.png")

# === Overlay Last 100 Hours Cost and Curtailment ===
# === Plot: Grid Import Cost – Last 100 Hours ===
save_timeseries_plot(
    df_last100['cost_rule'],
    df_last100['cost_cidt'],
    "Grid Import Cost (£)",
    "Grid Cost – Last 100 Hours",
    "Grid_Cost_Last100h.png",
    index=df_last100.index
)

# === Plot: Curtailment – Last 100 Hours ===
save_timeseries_plot(
    df_last100['curtail_rule'],
    df_last100['curtail_cidt'],
    "Curtailment (kWh)",
    "Curtailment – Last 100 Hours",
    "Curtailment_Last100h.png",
    index=df_last100.index
)

print("📈 All plots saved: cost, curtailment, SoC, physics deviation.")












import matplotlib.pyplot as plt
import os

# === Last 100 hours ===
df_last100 = df.tail(100)

# SoC limits
SOC_MIN = 15
SOC_MAX = 90

# === Create 2x2 Subplots ===
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.flatten()

# Subplot labels
labels = ['(a)', '(b)', '(c)', '(d)']

# === Battery SoC (Top-Left) ===
axs[0].plot(df_last100.index, df_last100['battery_soc_rule'], '--', label='Battery SoC (Rule-Based)', color='crimson')
axs[0].plot(df_last100.index, df_last100['battery_soc_cidt'], label='Battery SoC (CADT-PIML)', color='seagreen')
axs[0].axhline(SOC_MIN, linestyle=':', color='red', label='Min SoC (15%)')
axs[0].axhline(SOC_MAX, linestyle=':', color='darkred', label='Max SoC (90%)')
axs[0].set_ylabel("Battery SoC [%]", fontsize=16)
axs[0].set_xlabel("")
axs[0].tick_params(labelbottom=False)
axs[0].tick_params(labelsize=14)
axs[0].legend(fontsize=12)
axs[0].grid(True)
axs[0].text(0.5, 1.08, labels[0], transform=axs[0].transAxes,
            fontsize=14, fontweight='bold', ha='center', va='bottom')

# === Hydrogen SoC (Top-Right) ===
axs[1].plot(df_last100.index, df_last100['hydrogen_soc_rule'], '--', label='HSL (Rule-Based)', color='crimson')
axs[1].plot(df_last100.index, df_last100['hydrogen_soc_cidt'], label='HSL (CADT-PIML)', color='seagreen')
axs[1].axhline(SOC_MIN, linestyle=':', color='red', label='Min HSL (15%)')
axs[1].axhline(SOC_MAX, linestyle=':', color='darkred', label='Max HSL (90%)')
axs[1].set_ylabel("HSL [%]", fontsize=16)
axs[1].set_xlabel("")
axs[1].tick_params(labelbottom=False)
axs[1].tick_params(labelsize=14)
axs[1].legend(fontsize=12)
axs[1].grid(True)
axs[1].text(0.5, 1.08, labels[1], transform=axs[1].transAxes,
            fontsize=14, fontweight='bold', ha='center', va='bottom')

# === Grid Cost (Bottom-Left) ===
axs[2].plot(df_last100.index, df_last100['cost_rule'], '--', label='Grid Cost (Rule-Based)', color='crimson')
axs[2].plot(df_last100.index, df_last100['cost_cidt'], label='Grid Cost (CADT-PIML)', color='seagreen')
axs[2].set_ylabel("Grid Cost (£)", fontsize=16)
axs[2].set_xlabel("Date & Time", fontsize=16)
axs[2].tick_params(labelrotation=45, labelsize=14)
axs[2].legend(fontsize=12)
axs[2].grid(True)
axs[2].text(0.5, 1.08, labels[2], transform=axs[2].transAxes,
            fontsize=14, fontweight='bold', ha='center', va='bottom')

# === Curtailment (Bottom-Right) ===
axs[3].plot(df_last100.index, df_last100['curtail_rule'], '--', label='Curtailment (Rule-Based)', color='crimson')
axs[3].plot(df_last100.index, df_last100['curtail_cidt'], label='Curtailment (CADT-PIML)', color='seagreen')
axs[3].set_ylabel("Curtailment (kWh)", fontsize=16)
axs[3].set_xlabel("Date & Time", fontsize=16)
axs[3].tick_params(labelrotation=45, labelsize=14)
axs[3].grid(True)
# No legend
axs[3].text(0.5, 1.08, labels[3], transform=axs[3].transAxes,
            fontsize=14, fontweight='bold', ha='center', va='bottom')

# === Layout and Save ===
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Overlay_Combined_Last100h.png"), dpi=600)
plt.close()

print("✅ Final 2x2 subplot saved with centered labels above each subplot.")
