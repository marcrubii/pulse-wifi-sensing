"""Generador de CSI sintético para desarrollar y validar el pipeline.

Modela el canal WiFi como suma de trayectos (multipath):

    H(f, t) = sum_p  a_p * exp(-j 2*pi f * tau_p(t))  +  ruido

- Trayectos ESTÁTICOS (LOS + reflejos en paredes/muebles): amplitud y retardo
  constantes en el tiempo. Presentes siempre.
- Trayecto DINÁMICO (reflejo en un cuerpo que se mueve): su retardo tau(t)
  varía con el tiempo -> introduce fluctuaciones temporales en amplitud y fase.
  Solo presente en la clase "movimiento".

Con "quieto" el CSI apenas cambia entre paquetes (solo ruido térmico).
Con "movimiento" aparece una firma temporal-frecuencial que el modelo aprende.

El formato de salida imita al del ESP32-CSI-Tool: una matriz
(n_paquetes, n_subportadoras) de números complejos por ventana de captura.
"""

from __future__ import annotations

import numpy as np

# 802.11 20 MHz: ~64 subportadoras (el ESP32 reporta del orden de 52-64 útiles).
N_SUBCARRIERS = 64
# Frecuencias normalizadas de cada subportadora (índices centrados en 0).
_SUBCARRIER_IDX = np.arange(N_SUBCARRIERS) - N_SUBCARRIERS // 2


def _static_channel(rng: np.random.Generator, n_paths: int = 4) -> np.ndarray:
    """Respuesta en frecuencia de los trayectos estáticos: vector (N_SUBCARRIERS,) complejo."""
    delays = rng.uniform(0.0, 0.5, size=n_paths)          # retardos normalizados
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
    """Genera una ventana de CSI (n_packets, N_SUBCARRIERS) complejo.

    label = 0 -> quieto ; label = 1 -> movimiento.
    """
    if rng is None:
        rng = np.random.default_rng()

    H_static = _static_channel(rng)
    csi = np.tile(H_static, (n_packets, 1)).astype(complex)

    if label == 1:
        # Trayecto dinámico: retardo que deriva suavemente + una oscilación
        # (cuerpo moviéndose -> cambio Doppler/multipath a lo largo de la ventana).
        a_dyn = rng.uniform(0.3, 0.8) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        f_osc = rng.uniform(0.5, 3.0)                       # "velocidad" del movimiento
        phase0 = rng.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, n_packets)
        tau_t = 0.3 + 0.25 * np.sin(2 * np.pi * f_osc * t + phase0)  # retardo variable
        for k in range(n_packets):
            csi[k] += a_dyn * np.exp(-1j * 2 * np.pi * _SUBCARRIER_IDX * tau_t[k] / N_SUBCARRIERS)

    # Ruido térmico gaussiano complejo según SNR.
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
    """Crea un dataset de ventanas etiquetadas.

    Devuelve:
        X : (2*n_per_class, n_packets, N_SUBCARRIERS) complejo
        y : (2*n_per_class,) con 0=quieto, 1=movimiento
    """
    rng = np.random.default_rng(seed)
    windows, labels = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            windows.append(generate_window(label, n_packets, snr_db, rng))
            labels.append(label)
    X = np.stack(windows)
    y = np.array(labels)
    # Barajar.
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    X, y = make_dataset(n_per_class=5)
    print("X:", X.shape, X.dtype, "| y:", y.shape, "| clases:", np.bincount(y))


def generate_breathing(
    bpm: float = 15.0,
    n_packets: int = 3000,
    fs: float = 100.0,
    snr_db: float = 25.0,
    chest_gain: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """CSI de una persona QUIETA respirando (n_packets, N_SUBCARRIERS) complejo.

    El pecho es un trayecto dinámico DÉBIL cuyo retardo oscila a la frecuencia
    respiratoria f_b = bpm/60. No hay movimiento corporal grande: la firma es una
    única sinusoide lenta y de amplitud pequeña sobre el canal estático.
    """
    if rng is None:
        rng = np.random.default_rng()

    H_static = _static_channel(rng)
    csi = np.tile(H_static, (n_packets, 1)).astype(complex)

    f_b = bpm / 60.0                                   # Hz
    t = np.arange(n_packets) / fs                      # tiempo real (s)
    a_chest = chest_gain * np.exp(1j * rng.uniform(0, 2 * np.pi))
    tau0 = rng.uniform(0.2, 0.4)
    tau_t = tau0 + 0.03 * np.sin(2 * np.pi * f_b * t)  # micro-desplazamiento del pecho
    for k in range(n_packets):
        csi[k] += a_chest * np.exp(-1j * 2 * np.pi * _SUBCARRIER_IDX * tau_t[k] / N_SUBCARRIERS)

    sig_power = np.mean(np.abs(csi) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape)
    )
    return csi + noise
