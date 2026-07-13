# Step 1: read the file and isolate the CSI field
import numpy as np

def load_esp32_csv(path):
    filas_csi = []           # here we accumulate the "[...]" text of each row
    with open(path) as f:
        for linea in f:
            linea = linea.strip()
            if not linea.startswith("CSI_DATA"):
                continue                     # skip empty lines or headers
            crudo = linea.split("[")[1]      # everything after the "["
            crudo = crudo.split("]")[0]      # drop the trailing "]"
            filas_csi.append(crudo)
    return filas_csi   # for now we return the text, to inspect it


def fila_a_amplitud(texto_crudo):
    nums = np.array(texto_crudo.split(), dtype=float)  # "101 -48 5" -> [101, -48, 5, ...]
    pares = nums.reshape(-1, 2)        # (128,) -> (64, 2): each row = [imag, real]
    imag = pares[:, 0]
    real = pares[:, 1]
    amplitud = np.sqrt(real**2 + imag**2)   # complex magnitude, per subcarrier
    return amplitud                          # array of 64 values


def cargar_amplitudes(path):
    filas = load_esp32_csv(path)                       # raw text of each row
    matriz = np.array([fila_a_amplitud(f) for f in filas])  # stack -> (n_rows, 64)
    return matriz
