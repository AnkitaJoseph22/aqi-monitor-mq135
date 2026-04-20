import serial, serial.tools.list_ports, time

# List all ports
for p in serial.tools.list_ports.comports():
    print(f"  {p.device} — {p.description}")

PORT = 'COM11'   # ← change to your port
ser = serial.Serial(PORT, 9600, timeout=3)
print(f"\nConnected to {PORT}. Reading for 10 seconds...\n")
time.sleep(2)   # wait for Arduino to boot

start_time = time.time()
while time.time() - start_time < 10:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if line:
        print(repr(line))

ser.close()