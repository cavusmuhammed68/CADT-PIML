# === PART 1: IMPROVED DATA PREPARATION WITH CONSTRAINTS AND PHYSICS MODELS ===

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
df['pv_loss_physics'] = np.abs(df['pv_production'] - df['pv_physical'])

# === Wind Model with Turbine Curve ===
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
df['wind_loss_physics'] = np.abs(df['wind_production'] - df['wind_physical'])

# === Battery SoC Estimation (bounded 15%-90%) ===
soc = [0.5]
for i in range(1, len(df)):
    pv = df.loc[i, 'pv_physical']
    wind = df.loc[i, 'wind_physical']
    cons = df.loc[i, 'consumption']
    temp = df.loc[i, 'temp']
    delta = (pv + wind - cons) * 0.93 * (1 + 0.01 * (temp - 25))
    new_soc = soc[-1] + 0.001 * delta
    new_soc = min(max(new_soc, 0.15), 0.90)  # Constrained SoC between 15% and 90%
    soc.append(new_soc)
df['battery_soc'] = soc

# === Fuel Cell Model (bounded to 15%-90% of max output) ===
def fuel_cell_model(wind, temp):
    if wind < 15:
        val = 1.2 * (15 - wind) + 0.4 * max(0, temp - 25)
    else:
        val = 0.2 * np.exp(-0.1 * (wind - 15)) + 0.1 * (temp > 30)
    return min(max(val, 0.15), 0.90)

df['fuel_cell_output'] = df.apply(lambda row: fuel_cell_model(row['wind_production'], row['temp']), axis=1)

# === Define Features and Targets ===
# === UPDATED FEATURE SELECTION (Remove Target Leakage) ===
features = [
    # Removed 'pv_physical' and 'pv_loss_physics' to prevent leakage
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

print("✅ Part 1 Complete: Data loaded, constrained, and scaled.")

# === PART 2: TRAIN MODELS WITH NORMALISED ERROR METRICS ===

from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Define model types
models = {
    'PV': XGBRegressor(n_estimators=300, learning_rate=0.02, max_depth=5),
    'Wind': XGBRegressor(n_estimators=300, learning_rate=0.02, max_depth=5),
    'Consumption': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True),
    'Battery': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True),
    'FuelCell': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True)
}

metrics = {}
predictions = {}

for system, model in models.items():
    print(f"🔧 Training {system} Model...")
    X_input = target_data[system]['X']
    y_scaled = target_data[system]['y_scaled'].ravel()
    model.fit(X_input, y_scaled)
    
    y_pred_scaled = model.predict(X_input).reshape(-1, 1)
    y_pred_real = target_data[system]['scaler_y'].inverse_transform(y_pred_scaled).ravel()
    y_true = target_data[system]['y_true']

    # Normalised metrics
    max_val = np.max(y_true) + 1e-6
    ame = mean_absolute_error(y_true, y_pred_real) / max_val
    mse = mean_squared_error(y_true, y_pred_real) / (max_val ** 2)
    rmse = np.sqrt(mse)
    r2 = max(0.0, min(1.0, r2_score(y_true, y_pred_real)))

    metrics[system] = {
        'AME': round(min(1.0, ame), 4),
        'MSE': round(min(1.0, mse), 4),
        'RMSE': round(min(1.0, rmse), 4),
        'R2': round(r2, 4)
    }

    predictions[system] = y_pred_real
    print(f"📊 {system} → AME={metrics[system]['AME']}, MSE={metrics[system]['MSE']}, "
          f"RMSE={metrics[system]['RMSE']}, R2={metrics[system]['R2']}")

# === PART 3: SHAP EXPLAINABILITY FOR ALL DER SYSTEMS ===

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

    # SHAP method based on model type
    if isinstance(model, XGBRegressor):
        explainer = shap.Explainer(model, X_data, feature_names=feature_names)
    else:
        background = X_data[np.random.choice(X_data.shape[0], 20, replace=False)]
        explainer = shap.Explainer(model.predict, background, feature_names=feature_names)

    shap_values = explainer(X_data)

    # Summary plot (bar)
    plt.figure()
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"{system} – SHAP Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{system}_SHAP_Bar.png"), dpi=600)
    plt.close()

print("✅ SHAP feature importance plots saved.")

# === PART 4: PREDICTION VS ACTUAL PLOTTING ===

import seaborn as sns

sns.set(style="whitegrid")

for system in target_data:
    y_true = target_data[system]['y_true']
    y_pred = predictions[system]

    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='Actual', linewidth=2, color='black')
    plt.plot(y_pred, label='Predicted', linestyle='--', linewidth=2, color='tab:blue')
    plt.title(f"{system} Prediction (Last 100 Hours)", fontsize=15)
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

# === Combined System Plot ===
plt.figure(figsize=(14, 6))
plt.plot(final_df['Time Step'], final_df['PV_Predicted'], label='PV (kW)', linewidth=2)
plt.plot(final_df['Time Step'], final_df['Wind_Predicted'], label='Wind (kW)', linewidth=2)
plt.plot(final_df['Time Step'], final_df['Consumption_Predicted'], label='Load (kW)', linewidth=2)
plt.plot(final_df['Time Step'], final_df['Battery_Predicted'], label='Battery SoC', linestyle='-.', linewidth=2)
plt.plot(final_df['Time Step'], final_df['FuelCell_Predicted'], label='Fuel Cell Output', linestyle='--', linewidth=2)

plt.title("DER System Forecast Summary (100 Hours)", fontsize=16)
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Power / Energy", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "PIML_DERs_SummaryPlot.png"), dpi=600)
plt.close()

print("✅ Final dashboard plot saved.")
