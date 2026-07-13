"""Estimación de frecuencia respiratoria a partir de amplitud CSI limpia.

Una persona quieta que respira imprime una oscilación lenta y periódica
(~0.12-0.55 Hz = 7-33 rpm) en la amplitud del CSI. La estimamos buscando el pico
del espectro temporal en la banda respiratoria, PROMEDIANDO el espectro de todas
las subportadoras para robustez frente al ruido.
"""
from __future__ import annotations

import numpy as np

BAND_LO, BAND_HI = 0.12, 0.55        # Hz (~7-33 rpm): rango fisiológico con margen


def estimate_breathing(amp: np.ndarray, fs: float = 100.0,
                       band=(BAND_LO, BAND_HI), zero_pad: int = 4,
                       combine: str = "weighted"):
    """Estima BPM respiratorios de una ventana de amplitud limpia (n, n_sub).

    combine: 'mean'     -> promedio plano de subportadoras (baseline).
             'weighted' -> pondera cada subportadora por su prominencia
                           respiratoria en banda (baja el suelo de SNR usable).
    Devuelve (bpm, confianza).
    """
    x = amp - amp.mean(axis=0, keepdims=True)
    n = x.shape[0]
    nfft = n * zero_pad
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    P = np.abs(np.fft.rfft(x, n=nfft, axis=0)) ** 2      # (nfft/2+1, n_sub)
    m = (freqs >= band[0]) & (freqs <= band[1])

    if combine == "weighted":
        band_P = P[m]                                    # (n_band, n_sub)
        w = band_P.max(axis=0) / (band_P.mean(axis=0) + 1e-12)  # prominencia por subportadora
        w = w / (w.sum() + 1e-12)                        # normalizar pesos
        Pc = P @ w                                       # espectro combinado ponderado
    else:
        Pc = P.mean(axis=1)                              # promedio plano

    P_band, f_band = Pc[m], freqs[m]
    i = int(np.argmax(P_band))
    conf = P_band[i] / (P_band.mean() + 1e-12)
    return float(f_band[i] * 60.0), float(conf)
