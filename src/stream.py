"""Lectura y detección en tiempo real de CSI desde un CSV que crece en vivo.

El dia 16 'idf.py monitor | findstr CSI_DATA > captura.csv' va escribiendo el
fichero mientras nosotros lo leemos. Este modulo lee solo lo nuevo y decide.
"""
from __future__ import annotations

import numpy as np

from src.load_esp32 import fila_a_amplitud


class CsiTailer:
    """Lee incrementalmente un CSV de CSI que crece (como 'tail -f').

    Recuerda el offset en bytes y devuelve solo las filas nuevas y COMPLETAS.
    Guarda una linea parcial para completarla en la siguiente lectura.
    """

    def __init__(self, path: str):
        self.path = path
        self.offset = 0
        self.resto = ""            # linea a medias pendiente de la lectura previa

    def read_new(self) -> np.ndarray:
        """Devuelve (m, 64) con las filas nuevas (o (0, 64) si no hay)."""
        with open(self.path) as f:
            f.seek(self.offset)
            datos = f.read()
            self.offset = f.tell()

        datos = self.resto + datos
        lineas = datos.split("\n")
        self.resto = lineas.pop()          # la ultima puede estar incompleta

        filas = []
        for linea in lineas:
            linea = linea.strip()
            if not linea.startswith("CSI_DATA"):
                continue
            try:
                crudo = linea.split("[")[1].split("]")[0]
                a = fila_a_amplitud(crudo)
            except (IndexError, ValueError):
                continue                    # linea corrupta -> saltar
            if a.size == 64:
                filas.append(a)
        return np.stack(filas) if filas else np.empty((0, 64))


from src.features import window_features
from src.preprocess import lowpass
from src.breathing import estimate_breathing


class LiveMonitor:
    """Buffer deslizante + veredicto en vivo.

    El discriminante movimiento/quieto NO es un umbral a mano (movimiento y
    respiracion se confunden con una simple varianza): es el CLASIFICADOR ya
    entrenado (features -> modelo), que se inyecta. La respiracion se estima
    aparte sobre el buffer largo cuando la persona esta quieta.
    """

    def __init__(self, clf=None, mov_label: int = 1, fs: float = 100.0,
                 dur_s: float = 60.0, win_len: int = 128, umbral_resp: float = 2.0):
        self.clf = clf                 # pipeline sklearn ya entrenado (o None)
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
            return {"estado": "calentando", "n": n}
        # 1) movimiento? -> clasificador sobre la ventana reciente
        moviendo = False
        if self.clf is not None:
            feat = window_features(self.buf[-self.win_len:])[None, :]
            moviendo = self.clf.predict(feat)[0] == self.mov_label
        if moviendo:
            return {"estado": "MOVIMIENTO", "rpm": None, "conf": None, "n": n}
        # 2) quieto -> estimar respiracion sobre el buffer largo
        rpm, conf = estimate_breathing(lowpass(self.buf, fs=self.fs, fc=1.0), fs=self.fs)
        estado = f"QUIETO · respira {rpm:.0f} rpm" if conf > self.umbral_resp else "VACIO / sin senal"
        return {"estado": estado, "rpm": round(rpm, 1), "conf": round(conf, 1), "n": n}

import time
from datetime import datetime
def monitor_vivo(path: str, clf, fs: float = 100.0, refresco_s: float = 2.0,
                 duracion_s: float | None = None, dur_buffer_s: float = 60.0) -> None:
    """Lee el CSV en vivo y va imprimiendo el veredicto cada `refresco_s`.

    duracion_s=None -> corre indefinidamente (Ctrl+C para parar). El dia 16:
    lanzar la captura a `path` y ejecutar esto en paralelo.
    """
    tailer = CsiTailer(path)
    mon = LiveMonitor(clf=clf, fs=fs, dur_s=dur_buffer_s)
    t0 = time.time()
    while duracion_s is None or time.time() - t0 < duracion_s:
        mon.push(tailer.read_new())
        v = mon.veredicto()
        hora = datetime.now().strftime("%H:%M:%S")
        extra = f"  conf={v['conf']}" if v.get("conf") is not None else ""
        print(f"[{hora}] {v['estado']:26s} (buffer {v['n'] / fs:4.0f}s){extra}")
        time.sleep(refresco_s)
