"""
Preprocesado de amplitud CSI real (ESP32) antes de extraer features.

El CSI real trae suciedad que el sintétitico no: subportadoras nulas,
picos de paquetes corruptos y ruido de alta frecuencia. Este módulo la limpia
sobre el eje TEMPORAL, subportadora a subportadora, sin borrar la fluctuación
lenta (de aproximadamente 0.5 Hz - 3Hz) que es la firma del movimiento humano."""


from __future__ import annotations
import numpy as np

def drop_null_subcarriers(amp: np.ndarray, min_std: float = 1e-3):
    """Quita subportadoras casi constantes (guarda/piloto/muertas).

    amp: (n_paquetes, n_sub) amplitud real.
    Devuelve (amp_filtrada, indices_conservados).
    """
    std = amp.std(axis=0)               # variabilidad temporal de cada subportadora
    keep = std > min_std                # máscara: True = tiene señal
    return amp[:, keep], np.where(keep)[0]

def hampel_filter(amp: np.ndarray, win: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Sustituye picos impulsivos por la mediana local, subportadora a subportadora.

    win: tamaño de la ventana temporal (impar). n_sigmas: umbral en MADs.
    """
    amp = amp.astype(float).copy()
    n = amp.shape[0]
    k = win // 2
    # constante para que la MAD estime la std de una gaussiana
    c = 1.4826
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        ventana = amp[lo:hi]                          # (<=win, n_sub)
        med = np.median(ventana, axis=0)             # mediana local por subportadora
        mad = np.median(np.abs(ventana - med), axis=0)
        sigma = c * mad + 1e-8
        fila = amp[i]
        es_pico = np.abs(fila - med) > n_sigmas * sigma
        amp[i] = np.where(es_pico, med, fila)         # reemplaza solo donde hay pico
    return amp

from scipy.signal import butter, filtfilt


def lowpass(amp: np.ndarray, fs: float = 100.0, fc: float = 5.0, order: int = 4) -> np.ndarray:
    """Filtro paso-bajo Butterworth sobre el tiempo, subportadora a subportadora.

    fs: tasa de muestreo (paquetes/s). fc: frecuencia de corte (Hz).
    Deja pasar la banda del movimiento/respiración y corta el ruido rápido.
    """
    nyq = fs / 2.0                       # frecuencia de Nyquist
    wn = fc / nyq                        # corte normalizado (0..1)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, amp, axis=0)   # filtra a lo largo del tiempo, sin desfase


def preprocess_amplitude(
    amp: np.ndarray,
    fs: float = 100.0,
    fc: float = 5.0,
    hampel_win: int = 7,
    n_sigmas: float = 3.0,
    min_std: float = 1e-3,
): 
    """Pipeline de limpieza de amplitud CSI real: nulas -> anti-picos -> paso-bajo.

    amp: (n_paquetes, n_sub) amplitud cruda del ESP32.
    Devuelve (amp_limpia, indices_subportadoras_conservadas).
    """
    amp, keep = drop_null_subcarriers(amp, min_std=min_std) # quita subportadoras nulas
    amp = hampel_filter(amp, win=hampel_win, n_sigmas=n_sigmas) # quita picos impulsivos
    amp = lowpass(amp, fs=fs, fc=fc) # quita ruido de alta frecuencia
    return amp, keep