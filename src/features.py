"""Limpieza de CSI y extracción de características para clasificar movimiento.

Idea física: cuando alguien se mueve, la AMPLITUD del CSI de cada subportadora
fluctúa a lo largo del tiempo (multipath cambiante). Cuando todo está quieto,
la amplitud es casi constante (solo ruido). Por eso las features más
discriminativas son medidas de VARIABILIDAD TEMPORAL de la amplitud.

Trabajamos con amplitud (|H|) y no con fase cruda: la fase del CSI real viene
contaminada por desajustes de reloj (CFO/SFO) entre TX y RX y necesita
saneado adicional; la amplitud es mucho más robusta para detección de
movimiento y es el estándar de facto para esta primera etapa.
"""

from __future__ import annotations

import numpy as np


def to_amplitude(window: np.ndarray) -> np.ndarray:
    """CSI complejo (n_packets, n_sub) -> amplitud (n_packets, n_sub)."""
    return np.abs(window)


def clean_amplitude(amp: np.ndarray) -> np.ndarray:
    """Limpieza básica de la matriz de amplitud.

    1. Recorta outliers por subportadora (percentiles 1-99) para quitar
       picos espurios de paquetes corruptos.
    2. Expresa cada subportadora como FLUCTUACIÓN RELATIVA respecto a su nivel
       medio: (amp - mean) / mean. Esto quita el nivel estático (que depende
       de la geometría de la sala, no del movimiento) pero CONSERVA la
       magnitud de la fluctuación temporal, que es la señal que buscamos.
       Ojo: NO usamos z-score (dividir por la std) porque forzaría varianza
       unitaria y borraría precisamente esa información.
    """
    amp = amp.astype(float).copy()
    lo = np.percentile(amp, 1, axis=0)
    hi = np.percentile(amp, 99, axis=0)
    amp = np.clip(amp, lo, hi)
    mean = amp.mean(axis=0, keepdims=True) + 1e-8
    return (amp - mean) / mean


def window_features(window: np.ndarray) -> np.ndarray:
    """Convierte una ventana de CSI complejo en un vector de características.

    Combina estadísticos de variabilidad temporal (los que separan
    movimiento/quieto) más algunos descriptores globales.
    """
    amp = to_amplitude(window)
    amp_n = clean_amplitude(amp)

    # Variabilidad temporal por subportadora (clave: movimiento -> alta).
    temporal_std = amp_n.std(axis=0)              # (n_sub,)
    # Diferencia entre paquetes consecutivos (energía de cambio).
    diff_energy = np.mean(np.abs(np.diff(amp_n, axis=0)), axis=0)  # (n_sub,)

    feats = [
        temporal_std.mean(),
        temporal_std.max(),
        temporal_std.std(),
        diff_energy.mean(),
        diff_energy.max(),
        # Correlación media entre paquetes separados: quieto -> ~1, movimiento -> baja.
        _lag1_autocorr(amp_n),
        np.abs(amp).mean(),        # nivel medio de señal (descriptor global)
        np.abs(amp).std(),
    ]
    return np.array(feats, dtype=float)


def _lag1_autocorr(amp_n: np.ndarray) -> float:
    """Autocorrelación temporal media a lag 1 sobre subportadoras normalizadas."""
    a = amp_n[:-1]
    b = amp_n[1:]
    num = np.sum(a * b, axis=0)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0)) + 1e-8
    return float(np.mean(num / den))


FEATURE_NAMES = [
    "tstd_mean", "tstd_max", "tstd_std",
    "diff_mean", "diff_max",
    "lag1_autocorr", "amp_mean", "amp_std",
]


def extract_features(X: np.ndarray) -> np.ndarray:
    """Aplica window_features a un dataset (n_windows, n_packets, n_sub)."""
    return np.stack([window_features(w) for w in X])


if __name__ == "__main__":
    from synth_csi import make_dataset

    X, y = make_dataset(n_per_class=3)
    F = extract_features(X)
    print("Features:", F.shape, "| nombres:", FEATURE_NAMES)
    print("Media por clase:")
    for c in (0, 1):
        print(f"  clase {c}:", np.round(F[y == c].mean(axis=0), 3))

def sliding_windows(amp: np.ndarray, win_len: int = 128, hop: int = 64) -> np.ndarray:
    """Parte una captura continua (n, n_sub) en ventanas solapadas.

    Devuelve (n_ventanas, win_len, n_sub). hop = paso entre ventanas
    (hop < win_len => solapamiento; hop = win_len => sin solape).
    """
    n = amp.shape[0]
    if n < win_len:
        raise ValueError(f"captura de {n} muestras < win_len={win_len}")
    starts = range(0, n - win_len + 1, hop)
    return np.stack([amp[s:s + win_len] for s in starts])