"""Builds a trainable dataset from real captures in data/raw/.

Each CSV is a continuous capture labeled BY FILENAME
(vacio_01.csv -> 'vacio', mov_03.csv -> 'mov'). Per-file flow:
load amplitude -> spike removal -> low-pass -> windowing. Everything is stacked.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from src.load_esp32 import cargar_amplitudes
from src.preprocess import hampel_filter, lowpass
from src.features import sliding_windows


def etiqueta_de_nombre(path: str) -> str:
    """vacio_01.csv -> 'vacio' (first token before '_' or '.')."""
    base = os.path.basename(path)
    return base.split("_")[0].split(".")[0]

def build_dataset(raw_dir: str = "data/raw", fs: float = 107.0, fc: float = 5.0,
                  win_len: int = 128, hop: int = 64):
    """Returns (X, y, classes, groups).

    groups: (n_windows,) source recording of each window, e.g. "vacio_01".
    Needed for honest evaluation: train on some recordings, test on OTHERS.
    A random split leaks -- windows from one recording share the static channel.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no CSV files in {raw_dir}")

    Xs, nombres, grupos = [], [], []
    for p in paths:
        amp = cargar_amplitudes(p)          # (n, 64) raw amplitude
        amp = hampel_filter(amp)            # spike removal
        amp = lowpass(amp, fs=fs, fc=fc)    # low-pass (keeps the slow band)
        w = sliding_windows(amp, win_len=win_len, hop=hop)
        Xs.append(w)
        nombres += [etiqueta_de_nombre(p)] * len(w)
        grupos += [os.path.basename(p)[:-4]] * len(w)
        print(f"  {os.path.basename(p):24s} -> {len(w):4d} windows "
              f"[{etiqueta_de_nombre(p)}]")

    X = np.concatenate(Xs, axis=0)
    clases = sorted(set(nombres))
    idx = {c: i for i, c in enumerate(clases)}
    y = np.array([idx[n] for n in nombres])
    return X, y, clases, np.array(grupos)
