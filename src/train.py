"""Trains and evaluates a motion-vs-still classifier on CSI.

Full end-to-end pipeline:
    raw CSI -> features -> scaling -> classifier -> metrics.

Uses RandomForest (robust, no fine-tuning) as a baseline. The goal of Phase 0
is not to squeeze accuracy but to have the pipeline built and validated.
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
    # 1. CSI data (synthetic for now; to be replaced by real ESP32 data).
    X_csi, y = make_dataset(n_per_class=n_per_class, snr_db=snr_db, seed=seed)
    print(f"CSI: {X_csi.shape}  classes={np.bincount(y)}  SNR={snr_db} dB")

    # 2. Feature extraction.
    X = extract_features(X_csi)

    # 3. Train/test split.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # 4. Model.
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=seed),
    )
    clf.fit(X_tr, y_tr)

    # 5. Evaluation.
    y_pred = clf.predict(X_te)
    print("\n== Confusion matrix (rows=true, cols=pred) ==")
    print("        still  motion")
    cm = confusion_matrix(y_te, y_pred)
    for name, row in zip(("still ", "motion"), cm):
        print(f"{name}   {row[0]:5d} {row[1]:6d}")
    print("\n== Report ==")
    print(classification_report(y_te, y_pred, target_names=["still", "motion"]))

    # 6. Feature importance (what the model looks at).
    rf = clf.named_steps["randomforestclassifier"]
    order = np.argsort(rf.feature_importances_)[::-1]
    print("== Feature importance ==")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:15s} {rf.feature_importances_[i]:.3f}")

    return clf


if __name__ == "__main__":
    run()
