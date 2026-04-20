import serial
import json
import threading
import time
import numpy as np
import joblib
import datetime
import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from collections import deque

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MANUAL_PORT = 'COM11'
BAUD_RATE = 9600

required_files = [
    'aqi_model.pkl',
    'aqi_scaler.pkl',
    'aqi_features.pkl',
    'aqi_constants.pkl',
    'anomaly_model.pkl',
    'forecast_model.pkl'
]

for f in required_files:
    full_path = os.path.join(BASE_DIR, f)
    if not os.path.exists(full_path):
        print(f"ERROR: {f} not found. Run trainmodel.py first.")
        exit(1)

print("Loading models...")

model = joblib.load(os.path.join(BASE_DIR, 'aqi_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'aqi_scaler.pkl'))
FEATURE_COLS = joblib.load(os.path.join(BASE_DIR, 'aqi_features.pkl'))
CONST = joblib.load(os.path.join(BASE_DIR, 'aqi_constants.pkl'))
anomaly_model = joblib.load(os.path.join(BASE_DIR, 'anomaly_model.pkl'))
forecast_model = joblib.load(os.path.join(BASE_DIR, 'forecast_model.pkl'))

RL = CONST['RL']
VIN = CONST['VIN']
RO_CLEAN = CONST['RO_CLEAN']
CURVE_A = CONST['CURVE_A']
CURVE_B = CONST['CURVE_B']

ppm_buffer = deque(maxlen=5)
history = deque(maxlen=500)
latest = {}
lock = threading.Lock()


def raw_to_voltage(raw):
    return raw * (VIN / 1023.0)


def voltage_to_rs(vout):
    if vout < 0.01:
        return 999.0
    return RL * ((VIN / vout) - 1.0)


def rs_to_ppm(rs_value):
    ratio = rs_value / RO_CLEAN
    if ratio <= 0:
        return 0.0
    return max(0.0, CURVE_A * (ratio ** CURVE_B))


def ppm_to_aqi(ppm):
    ppm = max(0.0, float(ppm))

    if ppm <= 10:
        aqi = (ppm / 10.0) * 50.0
    elif ppm <= 20:
        aqi = 50.0 + ((ppm - 10.0) / 10.0) * 50.0
    elif ppm <= 30:
        aqi = 100.0 + ((ppm - 20.0) / 10.0) * 50.0
    elif ppm <= 40:
        aqi = 150.0 + ((ppm - 30.0) / 10.0) * 50.0
    elif ppm <= 60:
        aqi = 200.0 + ((ppm - 40.0) / 20.0) * 100.0
    else:
        aqi = 300.0 + ((ppm - 60.0) / 40.0) * 200.0

    return round(min(aqi, 500.0), 1)


def aqi_status_from_value(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Poor"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Severe"
    else:
        return "Hazardous"


def build_features(raw, rs_value, ppm):
    ppm_buffer.append(ppm)

    log_ppm = np.log1p(ppm)
    rolling_mean = float(np.mean(ppm_buffer))
    rolling_std = float(np.std(ppm_buffer))

    return [raw, rs_value, ppm, log_ppm, rolling_mean, rolling_std]


def make_reading(raw, time_sec, source):
    raw = max(0.0, min(1023.0, float(raw)))
    vout = raw_to_voltage(raw)
    rs_value = voltage_to_rs(vout)
    ppm = rs_to_ppm(rs_value)
    aqi = ppm_to_aqi(ppm)
    aqi_status = aqi_status_from_value(aqi)

    features = build_features(raw, rs_value, ppm)
    feat_scaled = scaler.transform([features])

    conf = float(np.max(model.predict_proba(feat_scaled)[0])) * 100

    anomaly = anomaly_model.predict(feat_scaled)[0]
    anomaly_status = "Anomaly 🚨" if anomaly == -1 else "Normal"

    future_ppm = None
    if len(ppm_buffer) == 5:
        future_ppm = float(forecast_model.predict([list(ppm_buffer)])[0])

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "time_sec": round(time_sec, 2),
        "gas1_raw": round(raw, 2),
        "voltage": round(vout, 3),
        "Rs": round(rs_value, 2),
        "ppm": round(ppm, 2),
        "aqi": aqi,
        "aqi_status": aqi_status,
        "confidence": round(conf, 1),
        "anomaly": anomaly_status,
        "future_ppm": round(future_ppm, 2) if future_ppm is not None else None,
        "source": source
    }


def serial_reader():
    global latest
    ser = None

    while True:
        try:
            if ser is not None and ser.is_open:
                ser.close()

            print(f"Connecting to {MANUAL_PORT}...")
            ser = serial.Serial(MANUAL_PORT, BAUD_RATE, timeout=2)
            time.sleep(2)
            print(f"Connected to {MANUAL_PORT}")

            while True:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except Exception as read_error:
                    print(f"Read error: {read_error}")
                    break

                if not line:
                    continue

                print("Received:", line)

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print("Bad JSON:", line)
                    continue

                raw = float(data.get('raw', 0))
                time_sec = float(data.get('time', 0))

                reading = make_reading(raw, time_sec, "live")

                with lock:
                    latest = reading
                    history.append(reading)

                print("Stored latest:", latest)

        except serial.SerialException as e:
            print(f"Serial error: {e}")
            try:
                if ser is not None and ser.is_open:
                    ser.close()
            except Exception:
                pass
            time.sleep(3)

        except Exception as e:
            print(f"Unexpected error: {e}")
            try:
                if ser is not None and ser.is_open:
                    ser.close()
            except Exception:
                pass
            time.sleep(3)


threading.Thread(target=serial_reader, daemon=True).start()


@app.route('/api/live')
def api_live():
    with lock:
        data = dict(latest)

    if not data:
        return jsonify({"error": "No data yet"}), 503

    return jsonify(data), 200


@app.route('/api/history')
def api_history():
    with lock:
        data = list(history)
    return jsonify(data), 200


@app.route('/')
def dashboard():
    return send_from_directory(BASE_DIR, 'index.html')


if __name__ == '__main__':
    print("\nAQI Smart Monitor Running...")
    print(f"Dashboard: http://127.0.0.1:5000")
    print(f"Live API:   http://127.0.0.1:5000/api/live")
    app.run(host='0.0.0.0', port=5000, debug=False)