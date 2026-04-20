import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


CLASS_NAMES = {
    0: "Clean Air",
    1: "Ammonia",
    2: "CO2",
    3: "Benzene",
    4: "Natural Gas",
    5: "Carbon Monoxide",
    6: "LPG"
}

FEATURE_COLS = [
    'Gas1', 'Gas2', 'Gas3', 'Gas4', 'Gas5', 'Gas6',
    'Gas1 PPM', 'Gas2 PPM', 'Gas3 PPM', 'Gas4 PPM', 'Gas5 PPM', 'Gas6 PPM'
]


print("=" * 55)
print("  AQI Monitor - Model Training")
print("=" * 55)


print("\n[STEP 1] Loading dataset...")

if not os.path.exists('MQ135SensorData.csv'):
    print("ERROR: MQ135SensorData.csv not found in this folder.")
    print("Copy your dataset CSV into the aqi-monitor folder first.")
    exit()

df = pd.read_csv('MQ135SensorData.csv')
print(f"  Loaded {len(df):,} rows")
print(f"  Columns found: {list(df.columns)}")


print("\n[STEP 2] Checking class distribution...")
counts = df['Class'].value_counts().sort_index()
for cls, count in counts.items():
    name = CLASS_NAMES.get(int(cls), f"Unknown")
    bar  = "█" * (count // 1500)
    print(f"  Class {cls}  {name:<20}: {count:>7,}  {bar}")


print("\n[STEP 3] Cleaning bad sensor readings...")
sensor_cols = ['Gas1', 'Gas2', 'Gas3', 'Gas4', 'Gas5', 'Gas6']
before = len(df)
bad_mask = (df[sensor_cols] < 0).any(axis=1)
df = df[~bad_mask].reset_index(drop=True)
print(f"  Removed {before - len(df):,} bad rows (negative sensor values)")
print(f"  Clean rows remaining: {len(df):,}")

ppm_cols = ['Gas1 PPM','Gas2 PPM','Gas3 PPM','Gas4 PPM','Gas5 PPM','Gas6 PPM']
df[ppm_cols] = df[ppm_cols].clip(lower=0)
print(f"  Negative PPM values clipped to 0")


print("\n[STEP 4] Preparing features and labels...")
X = df[FEATURE_COLS].values
y = df['Class'].values
print(f"  Feature matrix shape : {X.shape}")
print(f"  Label vector shape   : {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"  Training samples : {len(X_train):,}")
print(f"  Testing  samples : {len(X_test):,}")


print("\n[STEP 5] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"  Gas1 raw  - original mean: {scaler.mean_[0]:.1f}")
print(f"  Gas1 PPM  - original mean: {scaler.mean_[6]:.1f}")
print(f"  After scaling: all features centered around 0")


print("\n[STEP 6] Training Random Forest model...")
print("  This will take 60-90 seconds. Please wait...")
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("  Training complete!")


print("\n[STEP 7] Evaluating model accuracy...")
y_pred   = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n  Overall Accuracy: {accuracy:.2f}%")
print()
print(classification_report(
    y_test, y_pred,
    target_names=[CLASS_NAMES[i] for i in range(7)]
))

print("  Which features the model relies on most:")
importances = model.feature_importances_
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 200)
    print(f"    {feat:>12} : {imp:.4f}  {bar}")


print("\n[STEP 8] Saving model files...")
joblib.dump(model,        'aqi_model.pkl')
joblib.dump(scaler,       'aqi_scaler.pkl')
joblib.dump(FEATURE_COLS, 'aqi_features.pkl')

print("  Saved: aqi_model.pkl")
print("  Saved: aqi_scaler.pkl")
print("  Saved: aqi_features.pkl")
print()
print("=" * 55)
print("  Done. Run app.py next.")
print("=" * 55)
```

Press **Ctrl + S** to save the file.

---

### Step 11 — Run the script

Go back to Command Prompt. Make sure you are still inside the `aqi-monitor` folder. Type:
```
python train_model.py
```

Press Enter.

You will see output appearing step by step. The training step (Step 6) will take about 60–90 seconds and shows nothing during that time — that's normal, it's working. Wait for it.

When it finishes you should see something like:
```
[STEP 7] Evaluating model accuracy...

  Overall Accuracy: 97.43%

  Which features the model relies on most:
      Gas3 PPM : 0.1823  ████████████████████████████████████
      Gas1 PPM : 0.1654  █████████████████████████████████
      Gas2 PPM : 0.1521  ██████████████████████████████

[STEP 8] Saving model files...
  Saved: aqi_model.pkl
  Saved: aqi_scaler.pkl
  Saved: aqi_features.pkl

  Done. Run app.py next.