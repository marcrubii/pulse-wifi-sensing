# WiFi Sensing with CSI — camera-free presence & breathing detection

Detect **presence, motion and breathing** of a person from the perturbations the
human body causes in a WiFi signal (*Channel State Information*, CSI). No camera →
privacy by design. RF passes through drywall, so it works across walls.

> **What it is:** a DSP + ML pipeline that, from the CSI of a single WiFi link,
> tells you *someone is here / moving / still and breathing at X bpm*.
>
> **What it is not:** it does not reconstruct images or body pose. Skeleton
> estimation ("DensePose from WiFi") is deliberately out of scope.

## Demo

Estimating the **breathing rate of a motionless person** (the signal is buried in
the raw CSI; the DSP pulls it out, and the system *refuses* when nobody is there
instead of inventing a number):

![Breathing detection demo](figures/demo_respiracion.png)

**Honest evaluation** — robustness against noise (where it works and where it
stops working). The biggest lever is window length, not the model:

![Robustness vs SNR](figures/robustez_respiracion.png)

> Both figures are generated on **synthetic CSI with known ground truth** (to
> validate the algorithm while isolating the physics). Capture with real hardware
> (2× ESP32) is the next step.

## How it works

```
CSI CSV     ──►  preprocessing        ──►  { windowing + classifier   ──►  presence / motion
(ESP32)          (null subcarriers ·       { breathing (FFT + PCA)     ──►  respiratory rate
                  spike removal ·          real-time verdict (stream)
                  low-pass)
```

- **Preprocessing** ([src/preprocess.py](src/preprocess.py)): drops null
  subcarriers, crushes impulsive spikes (robust Hampel filter) and band-limits to
  the slow motion/breathing band (zero-phase Butterworth).
- **Breathing** ([src/breathing.py](src/breathing.py)): temporal FFT with
  zero-padding + weighted subcarrier combining + a **confidence** score
  (peak/band-mean) that prevents false positives in an empty room.
- **Motion classifier** ([src/features.py](src/features.py),
  [src/train.py](src/train.py)): temporal-variability features → RandomForest
  (baseline).
- **Real time** ([src/stream.py](src/stream.py)): reads the CSV as it grows, keeps
  a sliding buffer and emits a live verdict.

## Modules

| File | What it does |
|---|---|
| `src/synth_csi.py` | Generates synthetic CSI (motion & breathing) to validate without hardware |
| `src/load_esp32.py` | Loads the real ESP32-CSI-Tool CSV → amplitude matrix |
| `src/preprocess.py` | Cleanup: null subcarriers → spike removal → low-pass |
| `src/features.py` | Features + sliding windows |
| `src/breathing.py` | Respiratory-rate estimator + confidence |
| `src/dataset.py` | Builds a labeled dataset from `data/raw/*.csv` |
| `src/train.py` | Trains and evaluates the motion/still classifier |
| `src/stream.py` | Incremental reader + live 3-state monitor |
| `notebooks/02_demo_respiracion.ipynb` | Visual demo of breathing detection |

## Getting started

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Breathing demo (on synthetic data, no hardware needed):

```powershell
jupyter notebook notebooks/02_demo_respiracion.ipynb
```

End-to-end classification pipeline:

```powershell
$env:PYTHONPATH="."; python src/train.py
```

## Status & honesty

- **Done:** full pipeline (preprocessing, windowing, classifier, breathing
  estimator, live monitor), validated on synthetic CSI with known ground truth.
- **In progress:** capture with real hardware (2× ESP32-WROOM-32) and evaluation
  on own data.
- **Known, accepted limit:** CSI **does not generalize across rooms** (the static
  channel changes). Honest evaluation requires training on some recordings and
  testing on **different** ones of the same class; a naive cross-validation over a
  single recording inflates the results. For a new environment you recalibrate on
  site (record a few minutes per class and retrain).

## Context

The **IEEE 802.11bf (WLAN Sensing)** standard is already published; WiFi sensing
is an active field (presence, elderly care, smart buildings). This project draws
on the CSI-sensing line of work (activity and vital-sign detection such as
breathing) and favors **one well-solved, honestly-evaluated case** over covering
everything.

Full roadmap and development log (in Spanish):
[bitacora_wifi_sensing.md](bitacora_wifi_sensing.md).
