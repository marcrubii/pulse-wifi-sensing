"""Construye un dataset entrenable a partir de capturas reales en data/raw/.

Cada CSV es una captura continua etiquetada POR EL NOMBRE DE ARCHIVO
(vacio_01.csv -> 'vacio', mov_03.csv -> 'mov'). Flujo por archivo:
cargar amplitud -> anti-picos -> paso-bajo -> ventanear. Se apila todo.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from src.load_esp32 import cargar_amplitudes
from src.preprocess import hampel_filter, lowpass
from src.features import sliding_windows


def etiqueta_de_nombre(path: str) -> str:
    """vacio_01.csv -> 'vacio' (primer token antes de '_' o '.')."""
    base = os.path.basename(path)
    return base.split("_")[0].split(".")[0]


def build_dataset(raw_dir: str = "data/raw", fs: float = 100.0, fc: float = 5.0,
                  win_len: int = 128, hop: int = 64):
    """Devuelve (X, y, clases).

    X: (n_ventanas, win_len, n_sub)  |  y: (n_ventanas,) enteros
    clases: lista de nombres; y indexa en ella.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no hay CSV en {raw_dir}")

    Xs, nombres = [], []
    for p in paths:
        amp = cargar_amplitudes(p)          # (n, 64) amplitud cruda
        amp = hampel_filter(amp)            # anti-picos
        amp = lowpass(amp, fs=fs, fc=fc)    # paso-bajo (mantiene banda lenta)
        w = sliding_windows(amp, win_len=win_len, hop=hop)
        Xs.append(w)
        nombres += [etiqueta_de_nombre(p)] * len(w)
        print(f"  {os.path.basename(p):24s} -> {len(w):4d} ventanas "
              f"[{etiqueta_de_nombre(p)}]")

    X = np.concatenate(Xs, axis=0)
    clases = sorted(set(nombres))
    idx = {c: i for i, c in enumerate(clases)}
    y = np.array([idx[n] for n in nombres])
    return X, y, clases
