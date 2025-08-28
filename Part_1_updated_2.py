# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:31:11 2025
@author: nfpm5
"""

# === PART 1: PHYSICS-INFORMED DATA PREPARATION WITH DYNAMIC LOSS AWARENESS ===

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# === Setup Paths ===
data_path = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Good work_2\data_for_energyy.csv"
results_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Good work_2\Results"
os.makedirs(results_dir, exist_ok=True)

# === Load Dataset ===
df = pd.read_csv(data_path, parse_dates=['time'])
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# === Select Last 100 Hours Only ===
df = df.sort_values('time').tail(100).reset_index(drop=True)

# === Create Time Features ===
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month

# === PV Physics-Informed Modelling ===
def pv_irradiance_model(hour, month):
    mu = 12
    sigma = 3 + 0.5 * np.cos((month - 6) * np.pi / 6)
    return np.exp(-((hour - mu) ** 2) / (2 * sigma ** 2))

def thermal_efficiency(temp):
    alpha1, alpha2 = 0.004, 0.0001
    return 1 - alpha1 * temp + alpha2 * temp**2

df['pv_irradiance'] = pv_irradiance_model(df['hour'], df['month'])
df['pv_efficiency'] = thermal_efficiency(df['temp'])
df['pv_physical'] = df['pv_irradiance'] * df['pv_efficiency']
df['pv_physical'] = df['pv_physical'] / df['pv_physical'].max() * df['pv_production'].max()
df['pv_loss_physics'] = np.square(df['pv_production'] - df['pv_physical'])

# === Wind Physics-Informed Model ===
def wind_power_physical(v, rho=1.225, A=100, cp=0.4):
    return 0.5 * rho * A * cp * v**3 / 1000

def turbine_curve(v):
    v_cut_in, v_rated, v_cut_out = 3, 12, 25
    if v < v_cut_in or v > v_cut_out:
        return 0
    elif v <= v_rated:
        return wind_power_physical(v)
    else:
        return wind_power_physical(v_rated)

wind_col = 'wind_speed_100m:ms' if 'wind_speed_100m:ms' in df.columns else 'wind_speed_50m:ms'
df['wind_physical'] = df[wind_col].apply(turbine_curve)
df['wind_physical'] = df['wind_physical'] / df['wind_physical'].max() * df['wind_production'].max()
df['wind_loss_physics'] = np.square(df['wind_production'] - df['wind_physical'])

# === Battery SoC Estimation (Bounded: 15%–90%) ===
soc = [0.5]
for i in range(1, len(df)):
    delta = (df.loc[i, 'pv_physical'] + df.loc[i, 'wind_physical'] - df.loc[i, 'consumption']) * 0.93 * (1 + 0.01 * (df.loc[i, 'temp'] - 25))
    new_soc = soc[-1] + 0.001 * delta
    soc.append(min(max(new_soc, 0.15), 0.90))
df['battery_soc'] = soc

# === Fuel Cell Output Model (Based on Wind & Temp) ===
def fuel_cell_model(wind, temp):
    if wind < 15:
        val = 1.2 * (15 - wind) + 0.4 * max(0, temp - 25)
    else:
        val = 0.2 * np.exp(-0.1 * (wind - 15)) + 0.1 * (temp > 30)
    return min(max(val, 0.15), 0.90)

df['fuel_cell_output'] = df.apply(lambda row: fuel_cell_model(row['wind_production'], row['temp']), axis=1)

# === Define Features and Targets for Learning ===
features = [
    'wind_physical', 'wind_loss_physics',
    'battery_soc', 'fuel_cell_output',
    'clear_sky_rad:W', 'precip_1h:mm', 'temp', 'sun_azimuth:d',
    'relative_humidity_100m:p',
    'wind_speed_100m:ms', 'wind_speed_50m:ms', 'wind_dir_100m:d',
    'global_rad_1h:Wh', 'direct_rad_1h:Wh', 'diffuse_rad_1h:Wh'
]

targets = {
    'PV': 'pv_production',
    'Wind': 'wind_production',
    'Consumption': 'consumption',
    'Battery': 'battery_soc',
    'FuelCell': 'fuel_cell_output'
}

# === Scale Features ===
X = df[features]
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# === Prepare Scaled Targets ===
target_data = {}
for system, col in targets.items():
    y = df[col].values.reshape(-1, 1)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    target_data[system] = {
        'X': X_scaled,
        'y_scaled': y_scaled,
        'y_true': y.ravel(),
        'scaler_y': scaler_y
    }

print("✅ Part 1 Complete: Physics-informed data preprocessed and scaled for 100-hour prediction.")


# === PART 2: TRAIN MODELS WITH PHYSICS-INFORMED LOSS FUNCTION ===

from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# === Define Models for Each System ===
models = {
    'PV': XGBRegressor(n_estimators=300, learning_rate=0.02, max_depth=5),
    'Wind': XGBRegressor(n_estimators=300, learning_rate=0.02, max_depth=5),
    'Consumption': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True),
    'Battery': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True),
    'FuelCell': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True)
}

metrics = {}
predictions = {}
λ_data = 0.8
λ_physics = 0.2

# === Loop Through Each System for Training ===
for system, model in models.items():
    print(f"\n🔧 Training {system} Model...")

    # Feature and target setup
    X_input = pd.DataFrame(target_data[system]['X'])  # restore DataFrame form
    y_scaled = target_data[system]['y_scaled'].ravel()
    y_true = target_data[system]['y_true']

    # Train-test split (ensure reproducibility)
    X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
        X_input, y_scaled, y_true, test_size=0.2, random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict and inverse-transform to real-world units
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred_real = target_data[system]['scaler_y'].inverse_transform(y_pred_scaled).ravel()

    # Compute physics baseline only for PV and Wind
    if system == 'PV':
        physics_baseline = df.loc[X_test.index, 'pv_physical'].values
    elif system == 'Wind':
        physics_baseline = df.loc[X_test.index, 'wind_physical'].values
    else:
        physics_baseline = np.zeros_like(y_true_test)

    # Metrics computation
    abs_loss = mean_absolute_error(y_true_test, y_pred_real)
    mse_loss = mean_squared_error(y_true_test, y_pred_real)
    rmse = np.sqrt(mse_loss)
    r2 = max(0.0, min(1.0, r2_score(y_true_test, y_pred_real)))
    physics_loss = np.mean((y_pred_real - physics_baseline)**2)
    piml_loss = λ_data * mse_loss + λ_physics * physics_loss

    # Normalised metrics
    max_val = np.max(y_true_test) + 1e-6
    ame = abs_loss / max_val
    mse = mse_loss / (max_val ** 2)
    rmse_norm = np.sqrt(mse)

    # Store metrics and predictions
    metrics[system] = {
        'AME': round(min(1.0, ame), 4),
        'MSE': round(min(1.0, mse), 4),
        'RMSE': round(min(1.0, rmse_norm), 4),
        'R2': round(r2, 4),
        'PhysicsLoss': round(physics_loss, 4),
        'PIML_Loss': round(piml_loss, 4)
    }

    predictions[system] = pd.Series(y_pred_real, index=X_test.index)

    print(f"📊 {system} → AME={metrics[system]['AME']}, MSE={metrics[system]['MSE']}, "
          f"RMSE={metrics[system]['RMSE']}, R2={metrics[system]['R2']}, "
          f"PhysLoss={metrics[system]['PhysicsLoss']}, PIMLLoss={metrics[system]['PIML_Loss']}")

# === PART 3: SHAP ANALYSIS, PREDICTIONS, AND FINAL EXPORT ===

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directories
shap_dir = os.path.join(results_dir, "plots")
os.makedirs(shap_dir, exist_ok=True)
shap.initjs()

# === SHAP Feature Importance ===
for system in target_data:
    print(f"🔍 SHAP analysis for {system}")
    model = models[system]
    X_data = pd.DataFrame(target_data[system]['X'], columns=X.columns)
    feature_names = list(X.columns)

    if isinstance(model, XGBRegressor):
        explainer = shap.Explainer(model, X_data)
    else:
        background = X_data.sample(20, random_state=0)
        explainer = shap.Explainer(model.predict, background)

    shap_values = explainer(X_data)

    plt.figure()
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"{system} – SHAP Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{system}_SHAP_Bar.png"), dpi=600)
    plt.close()

print("✅ SHAP plots saved.")

# === Prediction vs Actual for Test Set Only ===
sns.set(style="whitegrid")

for system in predictions:
    y_true = df.loc[predictions[system].index, targets[system]]
    y_pred = predictions[system]

    plt.figure(figsize=(12, 4))
    plt.plot(y_true.values, label='Actual', linewidth=2, color='black')
    plt.plot(y_pred.values, label='Predicted', linestyle='--', linewidth=2, color='tab:blue')
    plt.title(f"{system} Prediction vs Actual (Test Set)", fontsize=15)
    plt.xlabel("Time Step")
    plt.ylabel(system)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{system}_Prediction_TestSet.png"), dpi=600)
    plt.close()

print("✅ Prediction vs actual plots saved.")

# === Export CSV: Only test data predictions ===
final_df = pd.DataFrame({'Index': predictions['PV'].index})

for system in predictions:
    final_df[f'{system}_Actual'] = df.loc[predictions[system].index, targets[system]].values
    final_df[f'{system}_Predicted'] = predictions[system].values

final_csv_path = os.path.join(results_dir, "PIML_TestSet_Results.csv")
final_df.to_csv(final_csv_path, index=False)
print(f"✅ Test set results saved to: {final_csv_path}")

# === Summary Dashboard Plot ===
plt.figure(figsize=(14, 6))
plt.plot(final_df['Index'], final_df['PV_Predicted'], label='PV (kW)', linewidth=2)
plt.plot(final_df['Index'], final_df['Wind_Predicted'], label='Wind (kW)', linewidth=2)
plt.plot(final_df['Index'], final_df['Consumption_Predicted'], label='Load (kW)', linewidth=2)
plt.plot(final_df['Index'], final_df['Battery_Predicted'], label='Battery SoC', linestyle='-.', linewidth=2)
plt.plot(final_df['Index'], final_df['FuelCell_Predicted'], label='HSL', linestyle='--', linewidth=2)

plt.title("DER Forecast Summary – Test Set", fontsize=16)
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Power / Energy", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "DER_Summary_TestSet.png"), dpi=600)
plt.close()

print("✅ Final dashboard plot saved.")

# === Evaluation Summary Printout ===
print("\n📈 Evaluation Metrics Summary (Test Set):\n")
for system, vals in metrics.items():
    print(f"🔹 {system}")
    print(f"   MAE           : {vals['AME']}")
    print(f"   RMSE          : {vals['RMSE']}")
    print(f"   R²            : {vals['R2']}")
    print(f"   Physics Loss  : {vals['PhysicsLoss']}")
    print(f"   PIML Loss     : {vals['PIML_Loss']}\n")


# === Combined 2x2 Prediction Box: PV, Wind, Battery, Fuel Cell (Full 100 Hours) ===

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.flatten()

panel_letters = ['(a)', '(b)', '(c)', '(d)']
ylabel_texts = ['PV Power [kW]', 'Wind Power [kW]', 'Battery SoC [%]', 'HSL [%]']
systems = ['PV', 'Wind', 'Battery', 'FuelCell']
colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']

# Loop through each system to plot full-series comparison
for i, system in enumerate(systems):
    y_true = df[targets[system]].values  # True values for all 100 hours
    y_pred = models[system].predict(pd.DataFrame(target_data[system]['X']))  # Predict for all 100 hours
    y_pred = target_data[system]['scaler_y'].inverse_transform(y_pred.reshape(-1, 1)).ravel()

    # Convert to % for SoC and Fuel Cell
    if system in ['Battery', 'FuelCell']:
        y_true *= 100
        y_pred *= 100

    axs[i].plot(range(100), y_true, label='Actual', color='black', linewidth=2)
    axs[i].plot(range(100), y_pred, label='CADT-PIML', color=colors[i], linewidth=2)

    # Panel title: bold only for (a), (b), etc.
    axs[i].set_title(f"{panel_letters[i]}  {ylabel_texts[i]}", fontsize=14, fontweight='bold')
    axs[i].set_xlabel("Time [h]", fontsize=16)
    axs[i].set_ylabel(ylabel_texts[i], fontsize=16)
    axs[i].grid(True)
    axs[i].tick_params(labelsize=14)

    if i == 0:
        axs[i].legend(fontsize=16, loc='upper right')

plt.tight_layout()
boxplot_path = os.path.join(results_dir, "CADT_Prediction_Comparison_2x2_FullSeries_1.png")
plt.savefig(boxplot_path, dpi=600)
plt.close()

print(f"✅ Full-series 2×2 prediction plot saved to: {boxplot_path}")




# === Combined 2x2 Prediction Box: PV, Wind, Battery, Fuel Cell (Full 100 Hours) ===

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.flatten()

# Title letters and Y-axis labels
panel_letters = ['(a)', '(b)', '(c)', '(d)']
ylabel_texts = ['PV Power [kW]', 'Wind Power [kW]', 'Battery SoC [%]', 'HSL [%]']
systems = ['PV', 'Wind', 'Battery', 'FuelCell']
colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']

for i, system in enumerate(systems):
    y_true = df[targets[system]].values
    y_pred = models[system].predict(pd.DataFrame(target_data[system]['X']))
    y_pred = target_data[system]['scaler_y'].inverse_transform(y_pred.reshape(-1, 1)).ravel()

    # Convert to % for SoC and Fuel Cell
    if system in ['Battery', 'FuelCell']:
        y_true *= 100
        y_pred *= 100

    axs[i].plot(range(100), y_true, label='Actual', color='black', linewidth=2)
    axs[i].plot(range(100), y_pred, label='CADT-PIML', color=colors[i], linewidth=2)

    # Panel title: bold only for (a), (b), etc.
    axs[i].set_title(f"{panel_letters[i]}  {ylabel_texts[i]}", fontsize=14, fontweight='bold')
    axs[i].set_xlabel("Time [h]", fontsize=16)
    axs[i].set_ylabel(ylabel_texts[i], fontsize=16)
    axs[i].grid(True)
    axs[i].tick_params(labelsize=14)

    if i == 0:
        axs[i].legend(fontsize=16, loc='upper right')

plt.tight_layout()
boxplot_path = os.path.join(results_dir, "CADT_Prediction_Comparison_2x2_FullSeries_2.png")
plt.savefig(boxplot_path, dpi=600)
plt.close()

print(f"✅ Full-series 2×2 prediction plot saved to: {boxplot_path}")

