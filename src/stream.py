"""Real-time reading and detection of CSI from a CSV that grows live.

On capture day, 'idf.py monitor | findstr CSI_DATA > capture.csv' keeps writing
the file while we read it. This module reads only what is new and decides.
"""
from __future__ import annotations

import numpy as np

from src.load_esp32 import fila_a_amplitud


class CsiTailer:
    """Incrementally reads a growing CSI CSV (like 'tail -f').

    Remembers the byte offset and returns only the new, COMPLETE rows.
    Keeps a partial line to complete it on the next read.
    """

    def __init__(self, path: str):
        self.path = path
        self.offset = 0
        self.resto = ""            # half-line pending from the previous read

    def read_new(self) -> np.ndarray:
        """Returns (m, 64) with the new rows (or (0, 64) if none)."""
        with open(self.path) as f:
            f.seek(self.offset)
            datos = f.read()
            self.offset = f.tell()

        datos = self.resto + datos
        lineas = datos.split("\n")
        self.resto = lineas.pop()          # the last one may be incomplete

        filas = []
        for linea in lineas:
            linea = linea.strip()
            if not linea.startswith("CSI_DATA"):
                continue
            try:
                crudo = linea.split("[")[1].split("]")[0]
                a = fila_a_amplitud(crudo)
            except (IndexError, ValueError):
                continue                    # corrupt line -> skip
            if a.size == 64:
                filas.append(a)
        return np.stack(filas) if filas else np.empty((0, 64))


from src.features import window_features
from src.preprocess import lowpass
from src.breathing import estimate_breathing


class LiveMonitor:
    """Sliding buffer + live verdict.

    The motion/still discriminator is NOT a hand-tuned threshold (motion and
    breathing get confused by a simple variance): it is the already-trained
    CLASSIFIER (features -> model), which is injected. Breathing is estimated
    separately over the long buffer when the person is still.
    """

    def __init__(self, clf=None, mov_label: int = 1, fs: float = 100.0,
                 dur_s: float = 60.0, win_len: int = 128, umbral_resp: float = 2.0):
        self.clf = clf                 # already-trained sklearn pipeline (or None)
        self.mov_label = mov_label
        self.fs = fs
        self.maxlen = int(dur_s * fs)
        self.win_len = win_len
        self.umbral_resp = umbral_resp
        self.buf = np.empty((0, 64))

    def push(self, filas: np.ndarray) -> None:
        if filas.size == 0:
            return
        self.buf = np.concatenate([self.buf, filas], axis=0)[-self.maxlen:]

    def veredicto(self) -> dict:
        n = len(self.buf)
        if n < max(self.win_len, int(2 * self.fs)):
            return {"state": "warming up", "n": n}
        # 1) motion? -> classifier over the recent window
        moviendo = False
        if self.clf is not None:
            feat = window_features(self.buf[-self.win_len:])[None, :]
            moviendo = self.clf.predict(feat)[0] == self.mov_label
        if moviendo:
            return {"state": "MOTION", "rpm": None, "conf": None, "n": n}
        # 2) still -> estimate breathing over the long buffer
        rpm, conf = estimate_breathing(lowpass(self.buf, fs=self.fs, fc=1.0), fs=self.fs)
        state = f"STILL · breathing {rpm:.0f} bpm" if conf > self.umbral_resp else "EMPTY / no signal"
        return {"state": state, "rpm": round(rpm, 1), "conf": round(conf, 1), "n": n}

import time
from datetime import datetime
def monitor_vivo(path: str, clf, fs: float = 100.0, refresco_s: float = 2.0,
                 duracion_s: float | None = None, dur_buffer_s: float = 60.0) -> None:
    """Reads the CSV live and prints the verdict every `refresco_s` seconds.

    duracion_s=None -> runs indefinitely (Ctrl+C to stop). On capture day:
    start the capture into `path` and run this in parallel.
    """
    tailer = CsiTailer(path)
    mon = LiveMonitor(clf=clf, fs=fs, dur_s=dur_buffer_s)
    t0 = time.time()
    while duracion_s is None or time.time() - t0 < duracion_s:
        mon.push(tailer.read_new())
        v = mon.veredicto()
        hora = datetime.now().strftime("%H:%M:%S")
        extra = f"  conf={v['conf']}" if v.get("conf") is not None else ""
        print(f"[{hora}] {v['state']:26s} (buffer {v['n'] / fs:4.0f}s){extra}")
        time.sleep(refresco_s)
