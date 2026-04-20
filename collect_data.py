import serial
import serial.tools.list_ports
import json
import csv
import datetime
import os
import time
import sys

BAUD_RATE    = 9600
OUTPUT_FILE  = 'my_sensor_data.csv'
SESSION_CLASS = 0

CLASS_NAMES = {
    0: "Clean Air",
    1: "Ammonia",
    2: "CO2",
    3: "Benzene",
    4: "Natural Gas",
    5: "Carbon Monoxide",
    6: "LPG"
}

CSV_COLUMNS = [
    'Date ', 'Time(sec)',
    'Gas1', 'Gas2', 'Gas3', 'Gas4', 'Gas5', 'Gas6',
    'Gas1 PPM', 'Gas2 PPM', 'Gas3 PPM', 'Gas4 PPM', 'Gas5 PPM', 'Gas6 PPM',
    'Class'
]


def find_arduino():
    ports = list(serial.tools.list_ports.comports())
    print("Scanning for Arduino...")
    for p in ports:
        print(f"  Found: {p.device} — {p.description}")
    for p in ports:
        if any(kw in p.description.upper()
               for kw in ['ARDUINO', 'CH340', 'CH341', 'USB SERIAL', 'ATMEGA']):
            print(f"  Arduino found on: {p.device}")
            return p.device
    if ports:
        return ports[0].device
    return None


def collect(port, class_label):
    today = datetime.date.today().strftime('%d/%m/%Y')
    gas_name = CLASS_NAMES.get(class_label, f"Class {class_label}")

    print(f"\nConnecting to {port}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
    except serial.SerialException as e:
        print(f"\nERROR: Could not open port {port}")
        print(f"Reason: {e}")
        print("\nFix: Close Arduino IDE Serial Monitor if open.")
        sys.exit(1)

    time.sleep(2)

    file_exists = os.path.exists(OUTPUT_FILE)
    csvfile = open(OUTPUT_FILE, 'a', newline='')
    writer  = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()
        print(f"Created: {OUTPUT_FILE}")
    else:
        print(f"Appending to: {OUTPUT_FILE}")

    print(f"\nRecording — Class {class_label} ({gas_name})")
    print("Press Ctrl+C to stop.\n")
    print(f"{'Time':>8}  {'Gas1 Raw':>10}  {'PPM':>6}  Rows")
    print("-" * 40)

    rows_saved = 0

    try:
        while True:
            try:
                line = ser.readline().decode('utf-8').strip()
            except UnicodeDecodeError:
                continue

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            time_sec = round(data.get('time',    0.0), 2)
            gas1_raw = round(data.get('Gas1',    0.0), 2)
            gas1_ppm = int(  data.get('ppm',     0))

            row = {
                'Date ':      today,
                'Time(sec)':  time_sec,
                'Gas1':       gas1_raw,
                'Gas2':       0.0,
                'Gas3':       0.0,
                'Gas4':       0.0,
                'Gas5':       0.0,
                'Gas6':       0.0,
                'Gas1 PPM':   gas1_ppm,
                'Gas2 PPM':   0,
                'Gas3 PPM':   0,
                'Gas4 PPM':   0,
                'Gas5 PPM':   0,
                'Gas6 PPM':   0,
                'Class':      class_label
            }

            writer.writerow(row)
            csvfile.flush()
            rows_saved += 1

            if rows_saved % 10 == 0:
                print(f"{time_sec:>8.2f}  {gas1_raw:>10.2f}  {gas1_ppm:>6}  {rows_saved}")

    except KeyboardInterrupt:
        print(f"\nStopped.")

    finally:
        csvfile.close()
        ser.close()
        print(f"\nSaved {rows_saved} rows to {OUTPUT_FILE}")


if __name__ == '__main__':
    print("=" * 45)
    print("  AQI Monitor — Data Collector")
    print("=" * 45)
    print(f"\nRecording class: {SESSION_CLASS} ({CLASS_NAMES[SESSION_CLASS]})")
    print("To change gas class — edit SESSION_CLASS at top of file")
    print()
    port = find_arduino()
    if not port:
        print("ERROR: No Arduino found. Plug it in first.")
        sys.exit(1)
    collect(port, SESSION_CLASS)
