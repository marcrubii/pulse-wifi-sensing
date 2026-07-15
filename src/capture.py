# Save CSI_DATA lines by reading the AP's serial port directly.
# Reading the port ourselves avoids idf_monitor's forced RTS reset (idf_monitor.py:393),
# so the AP keeps running and the sta never drops the link between captures.
# Usage: python <path>/capture.py <COMx> <out.csv> <seconds> <warmup>
import sys
import time
import serial

puerto = sys.argv[1]
destino = sys.argv[2]
duracion = float(sys.argv[3])
calentamiento = float(sys.argv[4])

ser = serial.Serial()
ser.port = puerto
ser.baudrate = 921600
ser.timeout = 1
ser.dtr = False   # asserting either of these pulls EN low and reboots the AP
ser.rts = False
ser.open()


def leer_linea():
    return ser.readline().decode("utf-8", errors="ignore").strip()


# 1) wait until the link is actually delivering CSI
print("esperando CSI...", flush=True)
while not leer_linea().startswith("CSI_DATA"):
    pass

# 2) drop the warm-up: time to leave the room (vacio) or get walking (mov)
t0 = time.time()
while time.time() - t0 < calentamiento:
    leer_linea()

# 3) record
print(f"GRABANDO {duracion:.0f} s -> {destino}", flush=True)
n = 0
t0 = time.time()
with open(destino, "w") as f:
    while time.time() - t0 < duracion:
        linea = leer_linea()
        if linea.startswith("CSI_DATA"):
            f.write(linea + "\n")
            n += 1

t = time.time() - t0
print(f"LISTO: {n} filas en {t:.1f} s = {n / t:.1f} filas/s", flush=True)
