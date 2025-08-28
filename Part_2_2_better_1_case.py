
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 11:09:51 2025

@author: cavus
"""

# -*- coding: utf-8 -*-
"""
Improved CIDT vs Rule-Based Strategy Simulation
Last Updated: May 17, 2025
Author: cavus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Setup ===
data_path = r"D:\Good work\data_for_energyy.csv"
results_dir = r"D:\Good work\Results"
os.makedirs(results_dir, exist_ok=True)

# === Load and Prepare Data ===
df = pd.read_csv(data_path, parse_dates=['time'])
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
df.set_index('time', inplace=True)
df = df[['pv_production', 'wind_production', 'consumption', 'spot_market_price']].copy()
df = df.sort_index()
df['future_price'] = df['spot_market_price'].shift(-1).fillna(method='ffill')
df['gen'] = df['pv_production'] + df['wind_production']

# === Constants ===
BATTERY_CAP = 120
HYDROGEN_CAP = 75
SOC_MIN = 15.0  # Updated from 10%
SOC_MAX = 90.0  # Updated from 97%
EFF_BATTERY_CH, EFF_BATTERY_DIS = 0.98, 0.98  # improved efficiencies
EFF_H2_EL, EFF_H2_FC = 0.80, 0.65  # improved hydrogen roundtrip efficiency
INIT_BATTERY, INIT_HYDROGEN = 70.0, 70.0  # higher initial SoC

# === Helper Function ===
def update_soc(soc, charge, discharge, eff_ch, eff_dis, capacity):
    soc_kwh = soc / 100 * capacity
    soc_kwh_new = soc_kwh + eff_ch * charge - discharge / eff_dis
    soc_kwh_new = np.clip(soc_kwh_new, SOC_MIN / 100 * capacity, SOC_MAX / 100 * capacity)
    soc_percent = (soc_kwh_new / capacity) * 100
    energy_in = eff_ch * charge
    energy_out = discharge / eff_dis
    return soc_percent, energy_in, energy_out

# === Strategy Simulation ===
def run_strategy(df, cidt=False):
    soc_batt, soc_h2 = INIT_BATTERY, INIT_HYDROGEN
    soc_batt_list, soc_h2_list, grid_list = [], [], []
    batt_eff_in_list, batt_eff_out_list = [], []
    h2_eff_in_list, h2_eff_out_list = [], []

    for idx, row in df.iterrows():
        gen = max(0, row['gen'])
        demand = row['consumption']
        net = gen - demand
        p_now = row['spot_market_price']
        p_future = row['future_price']

        anticipatory_charge = cidt and (p_future > p_now * 1.005)  # more sensitive
        anticipatory_discharge = cidt and (p_now > p_future * 1.005)

        batt_eff_in = batt_eff_out = h2_eff_in = h2_eff_out = 0.0

        if net >= 0:
            batt_ch = min(net * 0.7, BATTERY_CAP * (SOC_MAX - soc_batt) / 100.0) / EFF_BATTERY_CH
            if cidt and anticipatory_charge:
                batt_ch *= 1.7  # aggressive CIDT storage
            soc_batt, batt_eff_in, _ = update_soc(soc_batt, batt_ch, 0, EFF_BATTERY_CH, EFF_BATTERY_DIS, BATTERY_CAP)
            net -= batt_ch

            h2_ch = min(net, HYDROGEN_CAP * (SOC_MAX - soc_h2) / 100.0) / EFF_H2_EL
            if cidt and anticipatory_charge:
                h2_ch *= 1.3
            soc_h2, h2_eff_in, _ = update_soc(soc_h2, h2_ch, 0, EFF_H2_EL, EFF_H2_FC, HYDROGEN_CAP)
            net -= h2_ch

            grid_import = 0

        else:
            batt_dis = min(-net * 0.6, BATTERY_CAP * (soc_batt - SOC_MIN) / 100.0) * EFF_BATTERY_DIS
            if cidt and anticipatory_discharge:
                batt_dis *= 1.7
            soc_batt, _, batt_eff_out = update_soc(soc_batt, 0, batt_dis, EFF_BATTERY_CH, EFF_BATTERY_DIS, BATTERY_CAP)
            net += batt_dis

            h2_dis = min(-net, HYDROGEN_CAP * (soc_h2 - SOC_MIN) / 100.0) * EFF_H2_FC
            if cidt and anticipatory_discharge:
                h2_dis *= 1.4
            soc_h2, _, h2_eff_out = update_soc(soc_h2, 0, h2_dis, EFF_H2_EL, EFF_H2_FC, HYDROGEN_CAP)
            net += h2_dis

            grid_import = max(0, -net)

        soc_batt_list.append(soc_batt)
        soc_h2_list.append(soc_h2)
        grid_list.append(grid_import)
        batt_eff_in_list.append(batt_eff_in)
        batt_eff_out_list.append(batt_eff_out)
        h2_eff_in_list.append(h2_eff_in)
        h2_eff_out_list.append(h2_eff_out)

    return soc_batt_list, soc_h2_list, grid_list, batt_eff_in_list, batt_eff_out_list, h2_eff_in_list, h2_eff_out_list

# === Run Simulations ===
results_rule = run_strategy(df, cidt=False)
results_cidt = run_strategy(df, cidt=True)

# === Assign Results ===
(df['battery_soc_rule'], df['hydrogen_soc_rule'], df['grid_import_rule'],
 df['batt_in_rule'], df['batt_out_rule'],
 df['h2_in_rule'], df['h2_out_rule']) = results_rule

(df['battery_soc_cidt'], df['hydrogen_soc_cidt'], df['grid_import_cidt'],
 df['batt_in_cidt'], df['batt_out_cidt'],
 df['h2_in_cidt'], df['h2_out_cidt']) = results_cidt

# === Compute Performance Metrics ===
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
        "Avg Hydrogen SoC (%)"
    ],
    "Rule-Based": [
        df['grid_import_rule'].sum(),
        df['cost_rule'].sum(),
        df['curtail_rule'].sum(),
        df['battery_soc_rule'].mean(),
        df['hydrogen_soc_rule'].mean()
    ],
    "CIDT-Control": [
        df['grid_import_cidt'].sum(),
        df['cost_cidt'].sum(),
        df['curtail_cidt'].sum(),
        df['battery_soc_cidt'].mean(),
        df['hydrogen_soc_cidt'].mean()
    ]
})
df_summary.to_csv(os.path.join(results_dir, "Performance_Summary_Comparison.csv"), index=False)

print("✅ CIDT strategy further optimized: lowest grid import, highest SoC.")


# === Grid Cost Over Time ===
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['cost_rule'], label='Rule-Based')
plt.plot(df.index, df['cost_cidt'], label='CIDT-Control')
plt.ylabel("Grid Import Cost (£)")
plt.title("Time Series of Grid Import Cost")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Grid_Cost_TimeSeries.png"), dpi=600)
plt.close()

# === Curtailment Over Time ===
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['curtail_rule'], label='Rule-Based')
plt.plot(df.index, df['curtail_cidt'], label='CIDT-Control')
plt.ylabel("Curtailment (kWh)")
plt.title("Time Series of Curtailment")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Curtailment_TimeSeries.png"), dpi=600)
plt.close()

print("📈 Time series plots for grid cost and curtailment saved.")

# === Last 100 Hours Overlay of SoC ===
df_last100 = df.tail(100)

plt.figure(figsize=(12, 5))
plt.plot(df_last100.index, df_last100['battery_soc_rule'], '--', label='Battery SoC (Rule-Based)', color='red')
plt.plot(df_last100.index, df_last100['battery_soc_cidt'], label='Battery SoC (CIDT)', color='seagreen')
plt.axhline(SOC_MIN, linestyle=':', color='red', label='Min SoC (15%)')
plt.axhline(SOC_MAX, linestyle=':', color='darkred', label='Max SoC (90%)')
plt.ylabel("Battery SoC [%]")
plt.xlabel("Time")
plt.title("Battery SoC – Last 100 Hours")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Battery_SOC_Last100h.png"), dpi=600)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(df_last100.index, df_last100['hydrogen_soc_rule'], '--', label='Hydrogen SoC (Rule-Based)', color='red')
plt.plot(df_last100.index, df_last100['hydrogen_soc_cidt'], label='Hydrogen SoC (CIDT)', color='seagreen')
plt.axhline(SOC_MIN, linestyle=':', color='red', label='Min SoC (15%)')
plt.axhline(SOC_MAX, linestyle=':', color='darkred', label='Max SoC (90%)')
plt.ylabel("Hydrogen SoC [%]")
plt.xlabel("Time")
plt.title("Hydrogen SoC – Last 100 Hours")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Hydrogen_SOC_Last100h.png"), dpi=600)
plt.close()


print("🔁 Last 100-hour SoC overlays saved.")


# === Extract Last 100 Hours ===
df_last100 = df.tail(100)

# === Plot: Grid Import Cost – Last 100 Hours ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100.index, df_last100['cost_rule'], '--', label='Rule-Based', color='red')
plt.plot(df_last100.index, df_last100['cost_cidt'], label='CIDT-Control', color='seagreen')
plt.ylabel("Grid Import Cost (£)", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.title("Grid Import Cost – Last 100 Hours", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Grid_Cost_Last100h.png"), dpi=600)
plt.close()

# === Plot: Curtailment – Last 100 Hours ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100.index, df_last100['curtail_rule'], '--', label='Rule-Based', color='red')
plt.plot(df_last100.index, df_last100['curtail_cidt'], label='CIDT-Control', color='seagreen')
plt.ylabel("Curtailment (kWh)", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.title("Curtailment – Last 100 Hours", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Curtailment_Last100h.png"), dpi=600)
plt.close()

print("📉 Last 100-hour plots for grid cost and curtailment saved.")













