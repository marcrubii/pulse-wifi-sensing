"""Respiratory-rate estimation from clean CSI amplitude.

A still person who breathes imprints a slow, periodic oscillation
(~0.12-0.55 Hz = 7-33 bpm) on the CSI amplitude. We estimate it by finding the
peak of the temporal spectrum in the breathing band, AVERAGING the spectrum of
all subcarriers for robustness against noise.
"""
from __future__ import annotations

import numpy as np

BAND_LO, BAND_HI = 0.12, 0.55        # Hz (~7-33 bpm): physiological range with margin


def estimate_breathing(amp: np.ndarray, fs: float = 100.0,
                       band=(BAND_LO, BAND_HI), zero_pad: int = 4,
                       combine: str = "weighted"):
    """Estimate breathing BPM from a clean amplitude window (n, n_sub).

    combine: 'mean'     -> flat average of subcarriers (baseline).
             'weighted' -> weight each subcarrier by its breathing prominence
                           in band (lowers the usable SNR floor).
    Returns (bpm, confidence).
    """
    x = amp - amp.mean(axis=0, keepdims=True)
    n = x.shape[0]
    nfft = n * zero_pad
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    P = np.abs(np.fft.rfft(x, n=nfft, axis=0)) ** 2      # (nfft/2+1, n_sub)
    m = (freqs >= band[0]) & (freqs <= band[1])

    if combine == "weighted":
        band_P = P[m]                                    # (n_band, n_sub)
        w = band_P.max(axis=0) / (band_P.mean(axis=0) + 1e-12)  # prominence per subcarrier
        w = w / (w.sum() + 1e-12)                        # normalize weights
        Pc = P @ w                                       # weighted combined spectrum
    else:
        Pc = P.mean(axis=1)                              # flat average

    P_band, f_band = Pc[m], freqs[m]
    i = int(np.argmax(P_band))
    conf = P_band[i] / (P_band.mean() + 1e-12)
    return float(f_band[i] * 60.0), float(conf)
