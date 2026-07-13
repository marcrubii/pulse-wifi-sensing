"""
Preprocessing of real (ESP32) CSI amplitude before extracting features.

Real CSI carries dirt that the synthetic one does not: null subcarriers,
spikes from corrupt packets and high-frequency noise. This module cleans it
along the TIME axis, subcarrier by subcarrier, without erasing the slow
fluctuation (~0.5 Hz - 3 Hz) that is the signature of human motion."""


from __future__ import annotations
import numpy as np

def drop_null_subcarriers(amp: np.ndarray, min_std: float = 1e-3):
    """Drop near-constant subcarriers (guard/pilot/dead).

    amp: (n_packets, n_sub) real amplitude.
    Returns (filtered_amp, kept_indices).
    """
    std = amp.std(axis=0)               # temporal variability of each subcarrier
    keep = std > min_std                # mask: True = has signal
    return amp[:, keep], np.where(keep)[0]

def hampel_filter(amp: np.ndarray, win: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace impulsive spikes with the local median, subcarrier by subcarrier.

    win: temporal window size (odd). n_sigmas: threshold in MADs.
    """
    amp = amp.astype(float).copy()
    n = amp.shape[0]
    k = win // 2
    # constant so the MAD estimates the std of a Gaussian
    c = 1.4826
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        ventana = amp[lo:hi]                          # (<=win, n_sub)
        med = np.median(ventana, axis=0)             # local median per subcarrier
        mad = np.median(np.abs(ventana - med), axis=0)
        sigma = c * mad + 1e-8
        fila = amp[i]
        es_pico = np.abs(fila - med) > n_sigmas * sigma
        amp[i] = np.where(es_pico, med, fila)         # replace only where there is a spike
    return amp

from scipy.signal import butter, filtfilt


def lowpass(amp: np.ndarray, fs: float = 100.0, fc: float = 5.0, order: int = 4) -> np.ndarray:
    """Butterworth low-pass filter over time, subcarrier by subcarrier.

    fs: sampling rate (packets/s). fc: cutoff frequency (Hz).
    Passes the motion/breathing band and cuts the fast noise.
    """
    nyq = fs / 2.0                       # Nyquist frequency
    wn = fc / nyq                        # normalized cutoff (0..1)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, amp, axis=0)   # filter along time, zero phase


def preprocess_amplitude(
    amp: np.ndarray,
    fs: float = 100.0,
    fc: float = 5.0,
    hampel_win: int = 7,
    n_sigmas: float = 3.0,
    min_std: float = 1e-3,
):
    """Real CSI amplitude cleaning pipeline: nulls -> spike removal -> low-pass.

    amp: (n_packets, n_sub) raw ESP32 amplitude.
    Returns (clean_amp, kept_subcarrier_indices).
    """
    amp, keep = drop_null_subcarriers(amp, min_std=min_std) # drop null subcarriers
    amp = hampel_filter(amp, win=hampel_win, n_sigmas=n_sigmas) # remove impulsive spikes
    amp = lowpass(amp, fs=fs, fc=fc) # remove high-frequency noise
    return amp, keep
