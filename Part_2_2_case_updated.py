# === Setup ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_path = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Good work_2\data_for_energyy.csv"
results_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Good work_2\Results"
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

# === Constants (Physics-Informed Bounds) ===
BATTERY_CAP = 120          # [kWh]
HYDROGEN_CAP = 75          # [kWh]
SOC_MIN = 15.0             # [%]
SOC_MAX = 90.0             # [%]
EFF_BATTERY_CH = 0.98      # charging efficiency
EFF_BATTERY_DIS = 0.98     # discharging efficiency
EFF_H2_EL = 0.80           # electrolyser efficiency
EFF_H2_FC = 0.65           # fuel cell efficiency
INIT_BATTERY = 70.0        # [%]
INIT_HYDROGEN = 70.0       # [%]

# === Helper: SoC Update with Loss Return ===
def update_soc(soc, charge, discharge, eff_ch, eff_dis, capacity):
    soc_kwh = soc / 100 * capacity
    energy_in = eff_ch * charge
    energy_out = discharge / eff_dis
    soc_kwh_new = soc_kwh + energy_in - energy_out
    soc_kwh_new = np.clip(soc_kwh_new, SOC_MIN / 100 * capacity, SOC_MAX / 100 * capacity)
    soc_percent = (soc_kwh_new / capacity) * 100
    return soc_percent, energy_in, energy_out

# === Strategy Simulation (includes physics-informed losses) ===
def run_strategy(df, cidt=False):
    soc_batt, soc_h2 = INIT_BATTERY, INIT_HYDROGEN
    soc_batt_list, soc_h2_list, grid_list = [], [], []
    batt_eff_in_list, batt_eff_out_list = [], []
    h2_eff_in_list, h2_eff_out_list = [], []

    physics_losses = []
    grid_losses = []

    for idx, row in df.iterrows():
        gen = max(0, row['gen'])
        demand = row['consumption']
        net = gen - demand
        p_now = row['spot_market_price']
        p_future = row['future_price']

        anticipatory_charge = cidt and (p_future > p_now * 1.005)
        anticipatory_discharge = cidt and (p_now > p_future * 1.005)

        batt_eff_in = batt_eff_out = h2_eff_in = h2_eff_out = 0.0

        if net >= 0:
            batt_ch_limit = BATTERY_CAP * (SOC_MAX - soc_batt) / 100.0
            batt_ch = min(net * (1.0 if cidt else 0.7), batt_ch_limit) / EFF_BATTERY_CH
            if cidt and anticipatory_charge:
                batt_ch *= 1.2
            soc_batt, batt_eff_in, _ = update_soc(soc_batt, batt_ch, 0, EFF_BATTERY_CH, EFF_BATTERY_DIS, BATTERY_CAP)
            net -= batt_ch

            h2_ch_limit = HYDROGEN_CAP * (SOC_MAX - soc_h2) / 100.0
            h2_ch = min(net, h2_ch_limit) / EFF_H2_EL
            if cidt and anticipatory_charge:
                h2_ch *= 1.1
            soc_h2, h2_eff_in, _ = update_soc(soc_h2, h2_ch, 0, EFF_H2_EL, EFF_H2_FC, HYDROGEN_CAP)
            net -= h2_ch

            grid_import = 0

        else:
            batt_dis_limit = BATTERY_CAP * (soc_batt - SOC_MIN) / 100.0
            batt_dis = min(-net * (0.5 if cidt else 0.6), batt_dis_limit) * EFF_BATTERY_DIS
            if cidt and anticipatory_discharge:
                batt_dis *= 1.3
            soc_batt, _, batt_eff_out = update_soc(soc_batt, 0, batt_dis, EFF_BATTERY_CH, EFF_BATTERY_DIS, BATTERY_CAP)
            net += batt_dis

            h2_dis_limit = HYDROGEN_CAP * (soc_h2 - SOC_MIN) / 100.0
            h2_dis = min(-net, h2_dis_limit) * EFF_H2_FC
            if cidt and anticipatory_discharge:
                h2_dis *= 1.2
            soc_h2, _, h2_eff_out = update_soc(soc_h2, 0, h2_dis, EFF_H2_EL, EFF_H2_FC, HYDROGEN_CAP)
            net += h2_dis

            grid_import = max(0, -net)

        # === Physics-Informed Loss Function ===
        total_energy_supplied = gen + batt_eff_out + h2_eff_out + grid_import
        total_energy_used = demand + batt_eff_in + h2_eff_in
        balance_error = np.abs(total_energy_supplied - total_energy_used)
        grid_cost_penalty = grid_import * p_now

        physics_losses.append(balance_error)
        grid_losses.append(grid_cost_penalty)

        soc_batt_list.append(soc_batt)
        soc_h2_list.append(soc_h2)
        grid_list.append(grid_import)
        batt_eff_in_list.append(batt_eff_in)
        batt_eff_out_list.append(batt_eff_out)
        h2_eff_in_list.append(h2_eff_in)
        h2_eff_out_list.append(h2_eff_out)

    return (
        soc_batt_list, soc_h2_list, grid_list,
        batt_eff_in_list, batt_eff_out_list,
        h2_eff_in_list, h2_eff_out_list,
        physics_losses, grid_losses
    )

# === Run Simulations ===
results_rule = run_strategy(df, cidt=False)
results_cidt = run_strategy(df, cidt=True)

# === Unpack Results ===
(df['battery_soc_rule'], df['hydrogen_soc_rule'], df['grid_import_rule'],
 df['batt_in_rule'], df['batt_out_rule'],
 df['h2_in_rule'], df['h2_out_rule'],
 physics_losses_rule, grid_losses_rule) = results_rule

(df['battery_soc_cidt'], df['hydrogen_soc_cidt'], df['grid_import_cidt'],
 df['batt_in_cidt'], df['batt_out_cidt'],
 df['h2_in_cidt'], df['h2_out_cidt'],
 physics_losses_cidt, grid_losses_cidt) = results_cidt

# === Compute Performance Metrics ===
df['cost_rule'] = df['grid_import_rule'] * df['spot_market_price']
df['cost_cidt'] = df['grid_import_cidt'] * df['spot_market_price']
df['curtail_rule'] = (df['gen'] - df['consumption'] - df['grid_import_rule']).clip(lower=0)
df['curtail_cidt'] = (df['gen'] - df['consumption'] - df['grid_import_cidt']).clip(lower=0)

# === Physics-Informed Loss Summaries ===
physics_loss_rule_total = np.sum(physics_losses_rule)
physics_loss_cidt_total = np.sum(physics_losses_cidt)
grid_cost_rule_total = np.sum(grid_losses_rule)
grid_cost_cidt_total = np.sum(grid_losses_cidt)

# === Save Summary ===
df_summary = pd.DataFrame({
    "Metric": [
        "Total Grid Import (kWh)",
        "Total Grid Cost (£)",
        "Total Curtailment (kWh)",
        "Avg Battery SoC (%)",
        "Avg Hydrogen SoC (%)",
        "Physics Loss Total (kWh)",
        "Grid Cost Penalty (£)"
    ],
    "Rule-Based": [
        df['grid_import_rule'].sum(),
        df['cost_rule'].sum(),
        df['curtail_rule'].sum(),
        df['battery_soc_rule'].mean(),
        df['hydrogen_soc_rule'].mean(),
        physics_loss_rule_total,
        grid_cost_rule_total
    ],
    "CADT-PIML": [
        df['grid_import_cidt'].sum(),
        df['cost_cidt'].sum(),
        df['curtail_cidt'].sum(),
        df['battery_soc_cidt'].mean(),
        df['hydrogen_soc_cidt'].mean(),
        physics_loss_cidt_total,
        grid_cost_cidt_total
    ]
})

summary_path = os.path.join(results_dir, "Performance_Summary_Comparison.csv")
df_summary.to_csv(summary_path, index=False)
print(f"✅ Simulation complete. Results saved to: {summary_path}")

import matplotlib.pyplot as plt

# === Time Series: Grid Cost Over Time ===
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['cost_rule'], label='Rule-Based')
plt.plot(df.index, df['cost_cidt'], label='CADT-PIML')
plt.ylabel("Grid Import Cost (£)")
plt.title("Time Series of Grid Import Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Grid_Cost_TimeSeries.png"), dpi=600)
plt.close()

# === Time Series: Curtailment Over Time ===
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['curtail_rule'], label='Rule-Based')
plt.plot(df.index, df['curtail_cidt'], label='CIDT-PIML')
plt.ylabel("Curtailment (kWh)")
plt.title("Time Series of Curtailment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Curtailment_TimeSeries.png"), dpi=600)
plt.close()

print("📊 Time series plots saved.")

# === Last 100 Hours Subset ===
df_last100 = df.tail(100)

# === Overlay: Battery SoC ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100.index, df_last100['battery_soc_rule'], '--', label='Battery SoC (Rule-Based)', color='red')
plt.plot(df_last100.index, df_last100['battery_soc_cidt'], label='Battery SoC (CADT-PIML)', color='seagreen')
plt.axhline(SOC_MIN, linestyle=':', color='red', label='Min SoC (15%)')
plt.axhline(SOC_MAX, linestyle=':', color='darkred', label='Max SoC (90%)')
plt.ylabel("Battery SoC [%]")
plt.title("Battery SoC – Last 100 Hours")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Battery_SOC_Last100h.png"), dpi=600)
plt.close()

# === Overlay: Hydrogen SoC ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100.index, df_last100['hydrogen_soc_rule'], '--', label='HSL (Rule-Based)', color='red')
plt.plot(df_last100.index, df_last100['hydrogen_soc_cidt'], label='HSL (CADT-PIML)', color='seagreen')
plt.axhline(SOC_MIN, linestyle=':', color='red', label='Min SoC (15%)')
plt.axhline(SOC_MAX, linestyle=':', color='darkred', label='Max SoC (90%)')
plt.ylabel("HSL [%]")
plt.title("HSL – Last 100 Hours")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Hydrogen_HSL_Last100h.png"), dpi=600)
plt.close()

# === Overlay: Grid Cost – Last 100 Hours ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100.index, df_last100['cost_rule'], '--', label='Grid Cost (Rule-Based)', color='red')
plt.plot(df_last100.index, df_last100['cost_cidt'], label='Grid Cost (CADT-PIML)', color='seagreen')
plt.ylabel("Grid Cost (£)")
plt.title("Grid Import Cost – Last 100 Hours")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Grid_Cost_Last100h.png"), dpi=600)
plt.close()

# === Overlay: Curtailment – Last 100 Hours ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100.index, df_last100['curtail_rule'], '--', label='Curtailment (Rule-Based)', color='red')
plt.plot(df_last100.index, df_last100['curtail_cidt'], label='Curtailment (CADT-PIML)', color='seagreen')
plt.ylabel("Curtailment (kWh)")
plt.title("Curtailment – Last 100 Hours")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Curtailment_Last100h.png"), dpi=600)
plt.close()

print("✅ Last 100-hour comparisons completed and saved.")





















import matplotlib.pyplot as plt
import numpy as np

# Add hour index if not already present
df['hour'] = np.arange(len(df))

# Last 100 hours from 8660 to 8759
df_last100 = df[df['hour'] >= 8660].copy()

# ✅ Evenly spaced ticks across 100-hour range
last_ticks = [8660, 8685, 8710, 8735, 8759]

# === Battery SoC ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100['hour'], df_last100['battery_soc_rule'], '--', label='Battery SoC (Rule-Based)', color='red')
plt.plot(df_last100['hour'], df_last100['battery_soc_cidt'], label='Battery SoC (CADT-PIML)', color='seagreen')
plt.axhline(SOC_MIN, linestyle=':', color='red', label='Min SoC (15%)')
plt.axhline(SOC_MAX, linestyle=':', color='darkred', label='Max SoC (90%)')
plt.xlabel("Hour [h]", fontsize=16)
plt.ylabel("Battery SoC [%]", fontsize=16)
plt.title("Battery SoC – Last 100 Hours", fontsize=16)
plt.xticks(last_ticks, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Battery_SOC_Last100h.png"), dpi=600)
plt.close()

# === Hydrogen SoC ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100['hour'], df_last100['hydrogen_soc_rule'], '--', label='HSL (Rule-Based)', color='red')
plt.plot(df_last100['hour'], df_last100['hydrogen_soc_cidt'], label='HSL (CADT-PIML)', color='seagreen')
plt.axhline(SOC_MIN, linestyle=':', color='red', label='Min HSL (15%)')
plt.axhline(SOC_MAX, linestyle=':', color='darkred', label='Max HSL (90%)')
plt.xlabel("Hour [h]", fontsize=16)
plt.ylabel("HSL [%]", fontsize=16)
plt.title("HSL – Last 100 Hours", fontsize=16)
plt.xticks(last_ticks, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Hydrogen_HSL_Last100h.png"), dpi=600)
plt.close()

# === Grid Cost – Last 100 Hours ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100['hour'], df_last100['cost_rule'], '--', label='Grid Cost (Rule-Based)', color='red')
plt.plot(df_last100['hour'], df_last100['cost_cidt'], label='Grid Cost (CADT-PIML)', color='seagreen')
plt.xlabel("Hour [h]", fontsize=16)
plt.ylabel("Grid Cost (£)", fontsize=16)
plt.title("Grid Import Cost – Last 100 Hours", fontsize=16)
plt.xticks(last_ticks, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Grid_Cost_Last100h.png"), dpi=600)
plt.close()

# === Curtailment – Last 100 Hours ===
plt.figure(figsize=(12, 5))
plt.plot(df_last100['hour'], df_last100['curtail_rule'], '--', label='Curtailment (Rule-Based)', color='red')
plt.plot(df_last100['hour'], df_last100['curtail_cidt'], label='Curtailment (CADT-PIML)', color='seagreen')
plt.xlabel("Hour [h]", fontsize=16)
plt.ylabel("Curtailment (kWh)", fontsize=16)
plt.title("Curtailment – Last 100 Hours", fontsize=16)
plt.xticks(last_ticks, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Curtailment_Last100h.png"), dpi=600)
plt.close()

print("✅ Final corrected last 100-hour plots saved with even tick spacing.")










import matplotlib.pyplot as plt
import numpy as np
import os  # Ensure os is imported for os.path.join

# Add hour index if not already present
if 'hour' not in df.columns:
    df['hour'] = np.arange(len(df))

# Last 100 hours (from hour 8660 to 8759)
df_last100 = df[df['hour'] >= 8660].copy()

# Ensure the range is exactly 100 hours
df_last100 = df_last100[df_last100['hour'] <= 8759]

# Evenly spaced ticks across 100-hour range (start, end, 5 ticks)
last_ticks = np.linspace(8660, 8759, 5, dtype=int)

# Function for consistent plotting
def plot_line(y_rule, y_cidt, ylabel, title, filename, ylimits=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df_last100['hour'], df_last100[y_rule], '--', label=f'{ylabel} (Rule-Based)', color='red')
    plt.plot(df_last100['hour'], df_last100[y_cidt], label=f'{ylabel} (CADT)', color='seagreen')
    if ylimits:
        plt.axhline(ylimits[0], linestyle=':', color='red', label=f'Min {ylabel} ({ylimits[0]}%)')
        plt.axhline(ylimits[1], linestyle=':', color='darkred', label=f'Max {ylabel} ({ylimits[1]}%)')
    plt.xlabel("Hour [h]", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.xticks(last_ticks, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=600)
    plt.close()

# Define SoC limits
SOC_MIN = 15
SOC_MAX = 90

# === Battery SoC ===
plot_line(
    'battery_soc_rule', 'battery_soc_cidt',
    ylabel="Battery SoC [%]",
    title="Battery SoC – Last 100 Hours",
    filename="Battery_SOC_Last100h.png",
    ylimits=(SOC_MIN, SOC_MAX)
)

# === Hydrogen SoC ===
plot_line(
    'hydrogen_soc_rule', 'hydrogen_soc_cidt',
    ylabel="HSL [%]",
    title="HSL – Last 100 Hours",
    filename="Hydrogen_HSL_Last100h.png",
    ylimits=(SOC_MIN, SOC_MAX)
)

# === Grid Cost ===
plot_line(
    'cost_rule', 'cost_cidt',
    ylabel="Grid Cost (£)",
    title="Grid Import Cost – Last 100 Hours",
    filename="Grid_Cost_Last100h.png"
)

# === Curtailment ===
plot_line(
    'curtail_rule', 'curtail_cidt',
    ylabel="Curtailment (kWh)",
    title="Curtailment – Last 100 Hours",
    filename="Curtailment_Last100h.png"
)

print("✅ Final corrected last 100-hour plots saved with even tick spacing.")





import matplotlib.pyplot as plt
import numpy as np
import os  # Ensure os is imported for os.path.join

# Add hour index if not already present
if 'hour' not in df.columns:
    df['hour'] = np.arange(len(df))

# === Last 1000 hours ===
max_hour = df['hour'].max()
df_last1000 = df[df['hour'] >= max_hour - 999].copy()

# Evenly spaced x-ticks across 1000-hour range (e.g., 10 ticks)
last_ticks = np.linspace(df_last1000['hour'].min(), df_last1000['hour'].max(), 10, dtype=int)

# SoC limits
SOC_MIN = 15
SOC_MAX = 90

# === General plotting function ===
def plot_line(y_rule, y_cidt, ylabel, title, filename, ylimits=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df_last1000['hour'], df_last1000[y_rule], '--', label=f'{ylabel} (Rule-Based)', color='red')
    plt.plot(df_last1000['hour'], df_last1000[y_cidt], label=f'{ylabel} (CADT-PIML)', color='seagreen')
    
    if ylimits:
        plt.axhline(ylimits[0], linestyle=':', color='red', label=f'Min {ylabel} ({ylimits[0]}%)')
        plt.axhline(ylimits[1], linestyle=':', color='darkred', label=f'Max {ylabel} ({ylimits[1]}%)')
    
    plt.xlabel("Hour [h]", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.xticks(last_ticks, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=600)
    plt.close()

# === Battery SoC ===
plot_line(
    'battery_soc_rule', 'battery_soc_cidt',
    ylabel="Battery SoC [%]",
    title="Battery SoC – Last 1000 Hours",
    filename="Battery_SOC_Last1000h.png",
    ylimits=(SOC_MIN, SOC_MAX)
)

# === Hydrogen SoC ===
plot_line(
    'hydrogen_soc_rule', 'hydrogen_soc_cidt',
    ylabel="HSL [%]",
    title="HSL – Last 1000 Hours",
    filename="Hydrogen_HSL_Last1000h.png",
    ylimits=(SOC_MIN, SOC_MAX)
)

# === Grid Cost ===
plot_line(
    'cost_rule', 'cost_cidt',
    ylabel="Grid Cost (£)",
    title="Grid Import Cost – Last 1000 Hours",
    filename="Grid_Cost_Last1000h.png"
)

# === Curtailment ===
plot_line(
    'curtail_rule', 'curtail_cidt',
    ylabel="Curtailment (kWh)",
    title="Curtailment – Last 1000 Hours",
    filename="Curtailment_Last1000h.png"
)

print("✅ Final plots for the last 1000 hours saved with even x-axis tick spacing.")





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








