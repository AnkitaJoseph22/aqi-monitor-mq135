import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
import joblib
import os

# ════════════════════════════════════════════════════════════════
# MQ135 SENSOR CONSTANTS
# ════════════════════════════════════════════════════════════════
RL         = 10.0
VIN        = 5.0
RO_CLEAN   = 3.6

CURVE_A    = 116.6020682
CURVE_B    = -2.769034857

# AQI Labels
AQI_CLASS_NAMES = {
    0: "Good",
    1: "Moderate",
    2: "Poor",
    3: "Unhealthy",
    4: "Severe",
    5: "Hazardous"
}

print("=" * 60)
print("  SMART AQI MODEL TRAINING")
print("=" * 60)

# ── Step 1: Load dataset ────────────────────────────────────────
print("\n[1] Loading dataset...")

if os.path.exists('MQ135_cleaned.csv'):
    df = pd.read_csv('MQ135_cleaned.csv')
elif os.path.exists('MQ135SensorData.csv'):
    df = pd.read_csv('MQ135SensorData.csv')
    df.columns = df.columns.str.strip()
else:
    print("Dataset not found!")
    exit()

print(f"Loaded rows: {len(df):,}")

# ── Step 2: Calculate Rs and PPM ────────────────────────────────
print("\n[2] Calculating Rs and PPM...")

df['vout'] = df['Gas1'] * (VIN / 1023.0)

df['Rs'] = df['vout'].apply(
    lambda v: RL * ((VIN / v) - 1.0) if v > 0.01 else 999.0
)

df['ratio'] = df['Rs'] / RO_CLEAN
df['ppm'] = CURVE_A * (df['ratio'] ** CURVE_B)
df['ppm'] = df['ppm'].clip(lower=0)

# ── Step 3: Create AQI classes (SINGLE SENSOR LOGIC) ────────────
print("\n[3] Creating AQI classes from PPM...")

def ppm_to_aqi(ppm):
    if ppm <= 400:
        return 0
    elif ppm <= 1000:
        return 1
    elif ppm <= 2000:
        return 2
    elif ppm <= 5000:
        return 3
    elif ppm <= 10000:
        return 4
    else:
        return 5

df['aqi_class'] = df['ppm'].apply(ppm_to_aqi)

# ── Step 4: Feature Engineering ─────────────────────────────────
print("\n[4] Creating features...")

df['log_ppm'] = np.log1p(df['ppm'])
df['rolling_mean'] = df['ppm'].rolling(window=5).mean().bfill()
df['rolling_std'] = df['ppm'].rolling(window=5).std().fillna(0)

FEATURE_COLS = ['Gas1', 'Rs', 'ppm', 'log_ppm', 'rolling_mean', 'rolling_std']

X = df[FEATURE_COLS].values
y = df['aqi_class'].values

# ── Step 5: Train/Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Step 6: Scaling ─────────────────────────────────────────────
print("\n[5] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Step 7: AQI Classification Model ────────────────────────────
print("\n[6] Training AQI Model (Random Forest)...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\nAccuracy: {accuracy:.2f}%")
print(classification_report(y_test, y_pred))

# ── Step 8: Anomaly Detection ───────────────────────────────────
print("\n[7] Training Anomaly Model...")

anomaly_model = IsolationForest(
    contamination=0.02,
    random_state=42
)

anomaly_model.fit(X_train_scaled)

# ── Step 9: Forecast Model ──────────────────────────────────────
print("\n[8] Training Forecast Model...")

def create_sequences(data, window=5):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

ppm_values = df['ppm'].values
X_seq, y_seq = create_sequences(ppm_values, window=5)

forecast_model = RandomForestRegressor(n_estimators=100)
forecast_model.fit(X_seq, y_seq)

# ── Step 10: Save Everything ────────────────────────────────────
print("\n[9] Saving models...")

MODEL_CONSTANTS = {
    'RL': RL,
    'VIN': VIN,
    'RO_CLEAN': RO_CLEAN,
    'CURVE_A': CURVE_A,
    'CURVE_B': CURVE_B,
}

joblib.dump(model, 'aqi_model.pkl')
joblib.dump(scaler, 'aqi_scaler.pkl')
joblib.dump(FEATURE_COLS, 'aqi_features.pkl')
joblib.dump(MODEL_CONSTANTS, 'aqi_constants.pkl')
joblib.dump(anomaly_model, 'anomaly_model.pkl')
joblib.dump(forecast_model, 'forecast_model.pkl')

print("Saved:")
print("  aqi_model.pkl")
print("  anomaly_model.pkl")
print("  forecast_model.pkl")

print("\n✅ TRAINING COMPLETE")
print("=" * 60)