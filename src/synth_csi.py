"""Synthetic CSI generator to develop and validate the pipeline.

Models the WiFi channel as a sum of paths (multipath):

    H(f, t) = sum_p  a_p * exp(-j 2*pi f * tau_p(t))  +  noise

- STATIC paths (LOS + reflections off walls/furniture): amplitude and delay
  constant over time. Always present.
- DYNAMIC path (reflection off a moving body): its delay tau(t) varies over
  time -> introduces temporal fluctuations in amplitude and phase.
  Only present in the "motion" class.

When "still", the CSI barely changes between packets (thermal noise only).
When "motion", a time-frequency signature appears that the model learns.

The output format mimics the ESP32-CSI-Tool: a matrix
(n_packets, n_subcarriers) of complex numbers per capture window.
"""

from __future__ import annotations

import numpy as np

# 802.11 20 MHz: ~64 subcarriers (the ESP32 reports on the order of 52-64 usable).
N_SUBCARRIERS = 64
# Normalized frequency of each subcarrier (indices centered at 0).
_SUBCARRIER_IDX = np.arange(N_SUBCARRIERS) - N_SUBCARRIERS // 2


def _static_channel(rng: np.random.Generator, n_paths: int = 4) -> np.ndarray:
    """Frequency response of the static paths: complex (N_SUBCARRIERS,) vector."""
    delays = rng.uniform(0.0, 0.5, size=n_paths)          # normalized delays
    gains = rng.uniform(0.2, 1.0, size=n_paths) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_paths))
    H = np.zeros(N_SUBCARRIERS, dtype=complex)
    for a, tau in zip(gains, delays):
        H += a * np.exp(-1j * 2 * np.pi * _SUBCARRIER_IDX * tau / N_SUBCARRIERS)
    return H


def generate_window(
    label: int,
    n_packets: int = 128,
    snr_db: float = 25.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a CSI window (n_packets, N_SUBCARRIERS) complex.

    label = 0 -> still ; label = 1 -> motion.
    """
    if rng is None:
        rng = np.random.default_rng()

    H_static = _static_channel(rng)
    csi = np.tile(H_static, (n_packets, 1)).astype(complex)

    if label == 1:
        # Dynamic path: a slowly drifting delay + an oscillation
        # (moving body -> Doppler/multipath change along the window).
        a_dyn = rng.uniform(0.3, 0.8) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        f_osc = rng.uniform(0.5, 3.0)                       # "speed" of the motion
        phase0 = rng.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, n_packets)
        tau_t = 0.3 + 0.25 * np.sin(2 * np.pi * f_osc * t + phase0)  # varying delay
        for k in range(n_packets):
            csi[k] += a_dyn * np.exp(-1j * 2 * np.pi * _SUBCARRIER_IDX * tau_t[k] / N_SUBCARRIERS)

    # Complex Gaussian thermal noise according to SNR.
    sig_power = np.mean(np.abs(csi) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape)
    )
    return csi + noise


def make_dataset(
    n_per_class: int = 200,
    n_packets: int = 128,
    snr_db: float = 25.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a dataset of labeled windows.

    Returns:
        X : (2*n_per_class, n_packets, N_SUBCARRIERS) complex
        y : (2*n_per_class,) with 0=still, 1=motion
    """
    rng = np.random.default_rng(seed)
    windows, labels = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            windows.append(generate_window(label, n_packets, snr_db, rng))
            labels.append(label)
    X = np.stack(windows)
    y = np.array(labels)
    # Shuffle.
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    X, y = make_dataset(n_per_class=5)
    print("X:", X.shape, X.dtype, "| y:", y.shape, "| classes:", np.bincount(y))


def generate_breathing(
    bpm: float = 15.0,
    n_packets: int = 3000,
    fs: float = 100.0,
    snr_db: float = 25.0,
    chest_gain: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """CSI of a STILL breathing person (n_packets, N_SUBCARRIERS) complex.

    The chest is a WEAK dynamic path whose delay oscillates at the breathing
    frequency f_b = bpm/60. There is no large body motion: the signature is a
    single slow, small-amplitude sinusoid on top of the static channel.
    """
    if rng is None:
        rng = np.random.default_rng()

    H_static = _static_channel(rng)
    csi = np.tile(H_static, (n_packets, 1)).astype(complex)

    f_b = bpm / 60.0                                   # Hz
    t = np.arange(n_packets) / fs                      # real time (s)
    a_chest = chest_gain * np.exp(1j * rng.uniform(0, 2 * np.pi))
    tau0 = rng.uniform(0.2, 0.4)
    tau_t = tau0 + 0.03 * np.sin(2 * np.pi * f_b * t)  # chest micro-displacement
    for k in range(n_packets):
        csi[k] += a_chest * np.exp(-1j * 2 * np.pi * _SUBCARRIER_IDX * tau_t[k] / N_SUBCARRIERS)

    sig_power = np.mean(np.abs(csi) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape)
    )
    return csi + noise
