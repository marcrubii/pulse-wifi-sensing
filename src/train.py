"""Entrena y evalúa un clasificador movimiento vs quieto sobre CSI.

Pipeline completo end-to-end:
    CSI crudo -> features -> escalado -> clasificador -> métricas.

Usa RandomForest (robusto, sin ajuste fino) como línea base. El objetivo de
la Fase 0 no es exprimir la accuracy sino tener el pipeline montado y validado.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract_features
from synth_csi import make_dataset


def run(n_per_class: int = 200, snr_db: float = 25.0, seed: int = 0):
    # 1. Datos CSI (sintéticos por ahora; se sustituirán por datos reales ESP32).
    X_csi, y = make_dataset(n_per_class=n_per_class, snr_db=snr_db, seed=seed)
    print(f"CSI: {X_csi.shape}  clases={np.bincount(y)}  SNR={snr_db} dB")

    # 2. Extracción de características.
    X = extract_features(X_csi)

    # 3. Train/test split.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # 4. Modelo.
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=seed),
    )
    clf.fit(X_tr, y_tr)

    # 5. Evaluación.
    y_pred = clf.predict(X_te)
    print("\n== Matriz de confusión (filas=real, cols=pred) ==")
    print("        quieto  movim")
    cm = confusion_matrix(y_te, y_pred)
    for name, row in zip(("quieto", "movim "), cm):
        print(f"{name}   {row[0]:5d} {row[1]:6d}")
    print("\n== Reporte ==")
    print(classification_report(y_te, y_pred, target_names=["quieto", "movimiento"]))

    # 6. Importancia de features (qué mira el modelo).
    rf = clf.named_steps["randomforestclassifier"]
    order = np.argsort(rf.feature_importances_)[::-1]
    print("== Importancia de features ==")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:15s} {rf.feature_importances_[i]:.3f}")

    return clf


if __name__ == "__main__":
    run()
