"""CSI cleaning and feature extraction to classify motion.

Physical idea: when someone moves, the AMPLITUDE of each subcarrier's CSI
fluctuates over time (changing multipath). When everything is still, the
amplitude is almost constant (noise only). That is why the most discriminative
features are measures of the amplitude's TEMPORAL VARIABILITY.

We work with amplitude (|H|) and not raw phase: real CSI phase is contaminated
by clock mismatches (CFO/SFO) between TX and RX and needs extra sanitization;
amplitude is much more robust for motion detection and is the de-facto standard
for this first stage.
"""

from __future__ import annotations

import numpy as np


def to_amplitude(window: np.ndarray) -> np.ndarray:
    """Complex CSI (n_packets, n_sub) -> amplitude (n_packets, n_sub)."""
    return np.abs(window)


def clean_amplitude(amp: np.ndarray) -> np.ndarray:
    """Basic cleanup of the amplitude matrix.

    1. Clips outliers per subcarrier (percentiles 1-99) to remove spurious
       spikes from corrupt packets.
    2. Expresses each subcarrier as a RELATIVE FLUCTUATION about its mean
       level: (amp - mean) / mean. This removes the static level (which
       depends on room geometry, not on motion) but KEEPS the magnitude of
       the temporal fluctuation, which is the signal we are after.
       Note: we do NOT use a z-score (dividing by std) because it would force
       unit variance and erase exactly that information.
    """
    amp = amp.astype(float).copy()
    lo = np.percentile(amp, 1, axis=0)
    hi = np.percentile(amp, 99, axis=0)
    amp = np.clip(amp, lo, hi)
    mean = amp.mean(axis=0, keepdims=True) + 1e-8
    return (amp - mean) / mean


def window_features(window: np.ndarray) -> np.ndarray:
    """Turns a complex CSI window into a feature vector.

    Combines temporal-variability statistics (the ones that separate
    motion/still) plus a few global descriptors.
    """
    amp = to_amplitude(window)
    amp_n = clean_amplitude(amp)

    # Temporal variability per subcarrier (key: motion -> high).
    temporal_std = amp_n.std(axis=0)              # (n_sub,)
    # Difference between consecutive packets (energy of change).
    diff_energy = np.mean(np.abs(np.diff(amp_n, axis=0)), axis=0)  # (n_sub,)

    feats = [
        temporal_std.mean(),
        temporal_std.max(),
        temporal_std.std(),
        diff_energy.mean(),
        diff_energy.max(),
        # Mean correlation between adjacent packets: still -> ~1, motion -> low.
        _lag1_autocorr(amp_n),
        np.abs(amp).mean(),        # mean signal level (global descriptor)
        np.abs(amp).std(),
    ]
    return np.array(feats, dtype=float)


def _lag1_autocorr(amp_n: np.ndarray) -> float:
    """Mean lag-1 temporal autocorrelation over normalized subcarriers."""
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
    """Applies window_features to a dataset (n_windows, n_packets, n_sub)."""
    return np.stack([window_features(w) for w in X])


if __name__ == "__main__":
    from synth_csi import make_dataset

    X, y = make_dataset(n_per_class=3)
    F = extract_features(X)
    print("Features:", F.shape, "| names:", FEATURE_NAMES)
    print("Mean per class:")
    for c in (0, 1):
        print(f"  class {c}:", np.round(F[y == c].mean(axis=0), 3))

def sliding_windows(amp: np.ndarray, win_len: int = 128, hop: int = 64) -> np.ndarray:
    """Splits a continuous capture (n, n_sub) into overlapping windows.

    Returns (n_windows, win_len, n_sub). hop = step between windows
    (hop < win_len => overlap; hop = win_len => no overlap).
    """
    n = amp.shape[0]
    if n < win_len:
        raise ValueError(f"capture of {n} samples < win_len={win_len}")
    starts = range(0, n - win_len + 1, hop)
    return np.stack([amp[s:s + win_len] for s in starts])
