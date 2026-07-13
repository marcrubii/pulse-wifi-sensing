# Paso 1: Leer el archivo y aislar el campo del CSI
import numpy as np

def load_esp32_csv(path):
    filas_csi = []           # aquí iremos guardando el texto "[...]" de cada fila
    with open(path) as f:
        for linea in f:
            linea = linea.strip()
            if not linea.startswith("CSI_DATA"):
                continue                     # saltar líneas vacías o cabeceras
            crudo = linea.split("[")[1]      # todo lo que va tras el "["
            crudo = crudo.split("]")[0]      # quitar el "]" final
            filas_csi.append(crudo)
    return filas_csi   # de momento devolvemos el texto, para verlo


def fila_a_amplitud(texto_crudo):
    nums = np.array(texto_crudo.split(), dtype=float)  # "101 -48 5" -> [101, -48, 5, ...]
    pares = nums.reshape(-1, 2)        # (128,) -> (64, 2): cada fila = [imag, real]
    imag = pares[:, 0]
    real = pares[:, 1]
    amplitud = np.sqrt(real**2 + imag**2)   # módulo del complejo, por subportadora
    return amplitud                          # array de 64 valores


def cargar_amplitudes(path):
    filas = load_esp32_csv(path)                       # texto crudo de cada fila
    matriz = np.array([fila_a_amplitud(f) for f in filas])  # apilar -> (n_filas, 64)
    return matriz
