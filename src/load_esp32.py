
import numpy as np
SIG_MODE_COL = 5   # 0 = legacy (non-HT), 1 = HT

def load_esp32_csv(path, sig_mode=1):
    """Return the raw "[...]" text of each CSI row.

    Only rows matching `sig_mode` are kept: at the same RSSI, legacy and HT
    frames come out of the chip on different amplitude scales (~19 vs ~13), so
    mixing them injects 50% jumps that look like motion. sig_mode=None keeps all.
    """
    filas_csi = []
    with open(path) as f:
        for linea in f:
            linea = linea.strip()
            if not linea.startswith("CSI_DATA"):
                continue
            if sig_mode is not None and int(linea.split(",")[SIG_MODE_COL]) != sig_mode:
                continue
            crudo = linea.split("[")[1].split("]")[0]
            filas_csi.append(crudo)
    return filas_csi


def fila_a_amplitud(texto_crudo):
    nums = np.array(texto_crudo.split(), dtype=float)  # "101 -48 5" -> [101, -48, 5, ...]
    pares = nums.reshape(-1, 2)        # (128,) -> (64, 2): each row = [imag, real]
    imag = pares[:, 0]
    real = pares[:, 1]
    amplitud = np.sqrt(real**2 + imag**2)   # complex magnitude, per subcarrier
    return amplitud                          # array of 64 values

def cargar_amplitudes(path, sig_mode=1):
    
    filas = load_esp32_csv(path, sig_mode)
    matriz = np.array([fila_a_amplitud(f) for f in filas])
    return matriz
