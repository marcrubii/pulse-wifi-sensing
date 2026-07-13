# WiFi Sensing con CSI — detección de presencia y respiración sin cámara

Detecta **presencia, movimiento y respiración** de una persona a partir de las
perturbaciones que su cuerpo provoca en la señal WiFi (*Channel State
Information*, CSI). Sin cámara → enfoque de privacidad. El RF atraviesa tabiques,
así que funciona a través de paredes.

> **Qué es:** un pipeline de DSP + ML que, a partir del CSI de un enlace WiFi,
> dice *hay alguien / se mueve / está quieto respirando a X rpm*.
>
> **Qué no es:** no reconstruye imágenes ni poses. La estimación de esqueleto
> ("DensePose from WiFi") queda fuera de alcance a propósito.

## Demo

Detección de la **frecuencia respiratoria de una persona inmóvil** (la señal está
enterrada en el CSI crudo; el DSP la extrae y el sistema se *niega* cuando no hay
nadie, en vez de inventar un número):

![Demo de detección de respiración](figures/demo_respiracion.png)

**Evaluación honesta** — robustez frente al ruido (dónde funciona y dónde deja de
hacerlo). La palanca que más manda es la longitud de ventana, no el modelo:

![Curva de robustez vs SNR](figures/robustez_respiracion.png)

> Ambas figuras están generadas sobre **CSI sintético con verdad conocida** (para
> validar el algoritmo aislando la física). La captura con hardware real
> (2× ESP32) es el siguiente paso.

## Cómo funciona

```
CSV de CSI  ──►  preprocesado         ──►  { ventaneo + clasificador  ──►  presencia / movimiento
(ESP32)          (nulas · anti-picos       { respiración (FFT + PCA)   ──►  frecuencia respiratoria
                  · paso-bajo)             veredicto en vivo (stream)
```

- **Preprocesado** ([src/preprocess.py](src/preprocess.py)): quita subportadoras
  nulas, aplasta picos impulsivos (filtro Hampel robusto) y filtra a la banda
  lenta del movimiento/respiración (Butterworth de fase cero).
- **Respiración** ([src/breathing.py](src/breathing.py)): FFT temporal con
  zero-padding + combinación ponderada de subportadoras + una **confianza**
  (pico/media en banda) que evita falsos positivos en habitación vacía.
- **Clasificador de movimiento** ([src/features.py](src/features.py),
  [src/train.py](src/train.py)): características de variabilidad temporal →
  RandomForest (línea base).
- **Tiempo real** ([src/stream.py](src/stream.py)): lee el CSV mientras se llena,
  mantiene un buffer deslizante y emite el veredicto en vivo.

## Módulos

| Archivo | Qué hace |
|---|---|
| `src/synth_csi.py` | Genera CSI sintético (movimiento y respiración) para validar sin hardware |
| `src/load_esp32.py` | Carga el CSV real del ESP32-CSI-Tool → matriz de amplitud |
| `src/preprocess.py` | Limpieza: nulas → anti-picos → paso-bajo |
| `src/features.py` | Características + ventaneo deslizante |
| `src/breathing.py` | Estimador de frecuencia respiratoria + confianza |
| `src/dataset.py` | Construye dataset etiquetado desde `data/raw/*.csv` |
| `src/train.py` | Entrena y evalúa el clasificador movimiento/quieto |
| `src/stream.py` | Lectura incremental + monitor en vivo (3 estados) |
| `notebooks/02_demo_respiracion.ipynb` | Demo visual de la detección de respiración |

## Puesta en marcha

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Demo de respiración (sobre datos sintéticos, sin hardware):

```powershell
jupyter notebook notebooks/02_demo_respiracion.ipynb
```

Pipeline de clasificación end-to-end:

```powershell
$env:PYTHONPATH="."; python src/train.py
```

## Estado y honestidad

- **Hecho:** pipeline completo (preprocesado, ventaneo, clasificador, detector de
  respiración, monitor en vivo), validado sobre CSI sintético con verdad conocida.
- **En curso:** captura con hardware real (2× ESP32-WROOM-32) y evaluación sobre
  datos propios.
- **Límite conocido y asumido:** el CSI **no generaliza entre salas** (el canal
  estático cambia). La evaluación honesta exige entrenar en unas grabaciones y
  testear en **otras distintas** de la misma clase; una validación cruzada
  ingenua sobre una sola grabación infla los resultados. Para un entorno nuevo se
  recalibra in situ (grabar unos minutos por clase y reentrenar).

## Contexto

El estándar **IEEE 802.11bf (WLAN Sensing)** ya está publicado; el sensado por
WiFi es un área activa (presencia, cuidado de mayores, edificios inteligentes).
Este proyecto se inspira en la línea de trabajo de *sensing* por CSI (detección de
actividad y de signos vitales como la respiración) y prioriza **un caso bien
resuelto y honestamente evaluado** frente a abarcar todo.

Hoja de ruta completa y bitácora de desarrollo:
[bitacora_wifi_sensing.md](bitacora_wifi_sensing.md).
