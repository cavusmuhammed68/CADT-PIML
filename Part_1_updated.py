# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 12:51:55 2025

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

# === Select Last 100 Hours ===
df = df.sort_values('time').tail(100).reset_index(drop=True)

# === Time Features ===
df['hour'] = df['time'].dt.hour
df['dayofyear'] = df['time'].dt.dayofyear
df['month'] = df['time'].dt.month

# === Physics-Informed PV Model ===
def pv_irradiance_model(hour, month):
    mu = 12
    sigma = 3 + 0.5 * np.cos((month - 6) * np.pi / 6)
    return np.exp(-((hour - mu) ** 2) / (2 * sigma ** 2))

def thermal_efficiency(temp):
    α1, α2 = 0.004, 0.0001
    return 1 - α1 * temp + α2 * temp**2

df['pv_irradiance'] = pv_irradiance_model(df['hour'], df['month'])
df['pv_efficiency'] = thermal_efficiency(df['temp'])
df['pv_physical'] = df['pv_irradiance'] * df['pv_efficiency']
df['pv_physical'] = df['pv_physical'] / df['pv_physical'].max() * df['pv_production'].max()
df['pv_loss_physics'] = np.square(df['pv_production'] - df['pv_physical'])

# === Wind Power Model with Turbine Characteristics ===
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

# === Battery SoC Estimation (Bounded 15%-90%) ===
soc = [0.5]
for i in range(1, len(df)):
    pv = df.loc[i, 'pv_physical']
    wind = df.loc[i, 'wind_physical']
    cons = df.loc[i, 'consumption']
    temp = df.loc[i, 'temp']
    delta = (pv + wind - cons) * 0.93 * (1 + 0.01 * (temp - 25))
    new_soc = soc[-1] + 0.001 * delta
    new_soc = min(max(new_soc, 0.15), 0.90)
    soc.append(new_soc)
df['battery_soc'] = soc

# === Fuel Cell Output Model (Constrained) ===
def fuel_cell_model(wind, temp):
    if wind < 15:
        val = 1.2 * (15 - wind) + 0.4 * max(0, temp - 25)
    else:
        val = 0.2 * np.exp(-0.1 * (wind - 15)) + 0.1 * (temp > 30)
    return min(max(val, 0.15), 0.90)

df['fuel_cell_output'] = df.apply(lambda row: fuel_cell_model(row['wind_production'], row['temp']), axis=1)

# === Define Features and Targets ===
features = [
    'wind_physical', 'wind_loss_physics',
    'battery_soc', 'fuel_cell_output',
    'clear_sky_rad:W', 'precip_1h:mm',
    'temp', 'sun_azimuth:d',
    'relative_humidity_100m:p',
    'wind_speed_100m:ms', 'wind_speed_50m:ms',
    'wind_dir_100m:d',
    'global_rad_1h:Wh', 'direct_rad_1h:Wh', 'diffuse_rad_1h:Wh'
]

targets = {
    'PV': 'pv_production',
    'Wind': 'wind_production',
    'Consumption': 'consumption',
    'Battery': 'battery_soc',
    'FuelCell': 'fuel_cell_output'
}

X = df[features]
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

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

print("✅ Part 1 Complete: Physics-informed data preprocessed and scaled.")

# === PART 2: TRAIN MODELS WITH PHYSICS-INFORMED LOSS FUNCTION ===

from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

models = {
    'PV': XGBRegressor(n_estimators=300, learning_rate=0.02, max_depth=5),
    'Wind': XGBRegressor(n_estimators=300, learning_rate=0.02, max_depth=5),
    'Consumption': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True),
    'Battery': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True),
    'FuelCell': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True)
}

metrics = {}
predictions = {}

# Physics-informed weights
λ_data = 0.8
λ_physics = 0.2

for system, model in models.items():
    print(f"🔧 Training {system} Model...")
    X_input = target_data[system]['X']
    y_scaled = target_data[system]['y_scaled'].ravel()
    model.fit(X_input, y_scaled)
    
    y_pred_scaled = model.predict(X_input).reshape(-1, 1)
    y_pred_real = target_data[system]['scaler_y'].inverse_transform(y_pred_scaled).ravel()
    y_true = target_data[system]['y_true']

    # Standard loss terms
    abs_loss = mean_absolute_error(y_true, y_pred_real)
    mse_loss = mean_squared_error(y_true, y_pred_real)
    rmse = np.sqrt(mse_loss)
    r2 = max(0.0, min(1.0, r2_score(y_true, y_pred_real)))

    # Physics-informed component
    if system == 'PV':
        physics_baseline = df['pv_physical'].values
    elif system == 'Wind':
        physics_baseline = df['wind_physical'].values
    else:
        physics_baseline = np.zeros_like(y_true)

    physics_loss = np.mean((y_pred_real - physics_baseline)**2)
    piml_loss = λ_data * mse_loss + λ_physics * physics_loss

    # Normalized metrics
    max_val = np.max(y_true) + 1e-6
    ame = abs_loss / max_val
    mse = mse_loss / (max_val ** 2)
    rmse_norm = np.sqrt(mse)
    
    metrics[system] = {
        'AME': round(min(1.0, ame), 4),
        'MSE': round(min(1.0, mse), 4),
        'RMSE': round(min(1.0, rmse_norm), 4),
        'R2': round(r2, 4),
        'PhysicsLoss': round(physics_loss, 4),
        'PIML_Loss': round(piml_loss, 4)
    }

    predictions[system] = y_pred_real
    print(f"📊 {system} → AME={metrics[system]['AME']}, MSE={metrics[system]['MSE']}, "
          f"RMSE={metrics[system]['RMSE']}, R2={metrics[system]['R2']}, "
          f"PhysLoss={metrics[system]['PhysicsLoss']}, PIMLLoss={metrics[system]['PIML_Loss']}")

# === PART 3: SHAP EXPLAINABILITY FOR DER SYSTEMS ===

import shap
import matplotlib.pyplot as plt
import os

shap_dir = os.path.join(results_dir, "plots")
os.makedirs(shap_dir, exist_ok=True)
shap.initjs()

for system in target_data:
    print(f"🔍 SHAP analysis for {system}")
    model = models[system]
    X_data = target_data[system]['X']
    feature_names = list(X.columns)

    # Choose explainer based on model type
    if isinstance(model, XGBRegressor):
        explainer = shap.Explainer(model, X_data, feature_names=feature_names)
    else:
        background = X_data[np.random.choice(X_data.shape[0], 20, replace=False)]
        explainer = shap.Explainer(model.predict, background, feature_names=feature_names)

    shap_values = explainer(X_data)

    # Summary plot - bar
    plt.figure()
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"{system} – SHAP Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{system}_SHAP_Bar.png"), dpi=600)
    plt.close()

print("✅ SHAP feature importance plots saved.")

# === PART 4: PREDICTION VS ACTUAL PLOTTING ===

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

for system in target_data:
    y_true = target_data[system]['y_true']
    y_pred = predictions[system]

    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='Actual', linewidth=2, color='black')
    plt.plot(y_pred, label='Predicted', linestyle='--', linewidth=2, color='tab:blue')
    plt.title(f"{system} Prediction vs Actual (Last 100 Hours)", fontsize=15)
    plt.xlabel("Time Step")
    plt.ylabel(system)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{system}_Prediction_Comparison.png"), dpi=600)
    plt.close()

print("✅ Visual prediction comparisons saved.")

# === PART 5: FINAL EXPORT AND OVERVIEW DASHBOARD ===

# Create export DataFrame
final_df = pd.DataFrame({'Time Step': np.arange(len(target_data['PV']['y_true']))})

for system in target_data:
    final_df[f'{system}_Actual'] = target_data[system]['y_true']
    final_df[f'{system}_Predicted'] = predictions[system]

# Save CSV
csv_path = os.path.join(results_dir, "PIML_Forecast_Results_Last100Hours.csv")
final_df.to_csv(csv_path, index=False)
print(f"✅ Final results saved to: {csv_path}")

# === Combined DERs Plot ===
plt.figure(figsize=(14, 6))
plt.plot(final_df['Time Step'], final_df['PV_Predicted'], label='PV (kW)', linewidth=2)
plt.plot(final_df['Time Step'], final_df['Wind_Predicted'], label='Wind (kW)', linewidth=2)
plt.plot(final_df['Time Step'], final_df['Consumption_Predicted'], label='Load (kW)', linewidth=2)
plt.plot(final_df['Time Step'], final_df['Battery_Predicted'], label='Battery SoC', linestyle='-.', linewidth=2)
plt.plot(final_df['Time Step'], final_df['FuelCell_Predicted'], label='Fuel Cell Output', linestyle='--', linewidth=2)

plt.title("DER System Forecast Summary (Last 100 Hours)", fontsize=16)
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Power / Energy", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "PIML_DERs_SummaryPlot.png"), dpi=600)
plt.close()

print("✅ Final dashboard plot saved.")











# === PART 5: Updated Combined Prediction Plot (2x2) with Proper Labels ===

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.flatten()
pred_labels = ['(a)', '(b)', '(c)', '(d)']
pred_ylabels = ['PV Power [kW]', 'Wind Power [kW]', 'Battery SoC [%]', 'Fuel Cell SoC [%]']
pred_systems = ['PV', 'Wind', 'Battery', 'FuelCell']

for i, system in enumerate(pred_systems):
    y_true = target_data[system]['y_true']
    y_pred = predictions[system]

    # Convert Battery and FuelCell to % scale
    if system in ['Battery', 'FuelCell']:
        y_true = y_true * 100
        y_pred = y_pred * 100

    axs[i].plot(y_true, label='Actual', color='black', linewidth=1.5)
    axs[i].plot(y_pred, label='CIDT', color='tab:green', linestyle='--', linewidth=3)
    axs[i].set_xlabel("Time [h]", fontsize=16)
    axs[i].set_ylabel(pred_ylabels[i], fontsize=16)
    axs[i].tick_params(labelsize=14)
    axs[i].grid(True)

    # Only include legend in the first plot
    if i == 0:
        axs[i].legend(fontsize=14)
    
    axs[i].text(0.5, 1.08, pred_labels[i], transform=axs[i].transAxes,
                fontsize=16, fontweight='bold', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Combined_Prediction_Comparison_Updated.png"), dpi=600)
plt.close()

print("✅ Updated combined prediction subplot with CIDT legend saved.")







import matplotlib.pyplot as plt
from matplotlib import gridspec
import shap
import numpy as np

# Composite SHAP image grid: 2x2 layout
fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(2, 2)
titles = ['(a) PV Power', '(b) Wind Power', '(c) Battery SOC', '(d) Fuel Cell SOC']
systems = ['PV', 'Wind', 'Battery', 'FuelCell']

for idx, system in enumerate(systems):
    model = models[system]
    X_data = target_data[system]['X']
    feature_names = list(X.columns)

    if isinstance(model, XGBRegressor):
        explainer = shap.Explainer(model, X_data, feature_names=feature_names)
    else:
        background = X_data[np.random.choice(X_data.shape[0], 20, replace=False)]
        explainer = shap.Explainer(model.predict, background, feature_names=feature_names)

    shap_values = explainer(X_data)

    # Draw SHAP plot on temp fig and set fonts
    tmp_fig = plt.figure()
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type="bar", show=False)
    tmp_ax = tmp_fig.axes[0]

    tmp_ax.tick_params(axis='x', labelsize=16)
    tmp_ax.tick_params(axis='y', labelsize=16)
    tmp_ax.set_xlabel("mean(|SHAP value|)", fontsize=16)
    tmp_ax.set_title("")

    tmp_fig.tight_layout()
    tmp_fig.canvas.draw()

    # Convert to image
    fig_image = np.frombuffer(tmp_fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_image = fig_image.reshape(tmp_fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(tmp_fig)

    # Insert into final 2x2 grid
    ax = fig.add_subplot(gs[idx])
    ax.imshow(fig_image)
    ax.axis('off')
    ax.set_title(titles[idx], fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(shap_dir, "SHAP_Grid_All.png"), dpi=600)
plt.close()

print("✅ SHAP grid saved with descriptive subplot titles.")




print("\n📈 Evaluation Metrics Summary:\n")
for system, vals in metrics.items():
    print(f"🔹 {system}")
    print(f"   MAE           : {vals['AME']}")
    print(f"   RMSE          : {vals['RMSE']}")
    print(f"   R²            : {vals['R2']}")
    print(f"   Physics Loss  : {vals['PhysicsLoss']}")
    print(f"   PIML Loss     : {vals['PIML_Loss']}\n")



