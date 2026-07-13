# Proyecto: WiFi Sensing (sensado sin cámara con CSI)

> Bitácora viva. Guarda este archivo en tu Proyecto de Claude.
> Cada vez que retomemos, Claude lee esto y sabe exactamente dónde estamos.
> Actualiza las casillas [ ] → [x] y añade notas al final de cada sesión.

---

## 1. Visión y alcance (leer siempre antes de empezar)

**Qué construyo:** un sistema que detecta presencia, movimiento y actividad
humana usando las perturbaciones que un cuerpo provoca en las señales WiFi
(Channel State Information, CSI). Funciona a través de paredes de tabique
porque el RF las atraviesa. Sin cámara → ángulo de privacidad.

**Qué NO es (para no frustrarme):** no es una cámara de rayos X. No reconstruye
imágenes ni caras. La estimación de pose del esqueleto ("DensePose from WiFi")
es el moonshot de investigación, NO el objetivo del verano.

**Objetivo realista del verano:** un demo que funciona de verdad de
detección de presencia + actividad (idealmente detección de caídas, por el
ángulo de cuidado de mayores), documentado y publicado.

**Mi ventaja injusta:** soy de telecos → DSP + ML a la vez. Casi nadie con mi
perfil y edad tiene este proyecto.

**Contexto de mercado (2026):** el estándar IEEE 802.11bf (WLAN Sensing) ya
está publicado; 2026 es el año de los primeros pilotos comerciales en edificios
inteligentes, seguridad y cuidado de mayores. Llego a tiempo.

---

## 2. Roadmap (10 semanas)

### Fase 0 — Entender el terreno (Semana 1) · SIN hardware, SIN gastar
Objetivo: intuición del CSI y pipeline de ML montado con datos existentes.
- [x] Entender qué es CSI vs RSSI (por qué CSI da mucha más info)
- [x] Montar entorno Python (numpy, pandas, scipy, scikit-learn, matplotlib)
- [~] Conseguir un dataset público de CSI y visualizarlo (amplitud/fase)
      → hecho con CSI SINTÉTICO (multipath+ruido); dataset real ESP32 pendiente
- [x] LOADER del formato real ESP32-CSI-Tool escrito y validado
      → src/load_esp32.py (CSV real → matriz amplitud (n, 64)); probado contra
      el example_csi.csv oficial del repo y conectado al pipeline (features.py
      lo traga sin tocar nada). notebooks/01_loader_esp32.ipynb
- [x] Entrenar un clasificador simple: "hay movimiento" vs "quieto"
- [x] Entregable: notebook que carga CSI, lo limpia y clasifica algo básico
      → notebooks/00_fase0_pipeline_csi.ipynb

### Fase 1 — Hardware y captura propia (Semanas 2-3)
Objetivo: capturar mi propio CSI en casa.
- [x] Comprar 2× ESP32 (pedidas; llegan el 14/07). Placa: ESP32-WROOM-32
      DevKitC USB-C, chip serie CH340C.
- [~] Flashear el ESP32-CSI-Tool (uno TX, uno RX)
      → ENTORNO YA LISTO sin placa: ESP-IDF v4.3.6 instalado (Offline Installer),
      repo clonado en Desktop\ESP32-CSI-Tool, driver CH340 instalado. LOS DOS
      firmwares COMPILAN limpio: active_ap y active_sta (idf.py set-target esp32 +
      build OK). Falta solo el flasheo físico el día 14 (idf.py -p COMx flash).
- [ ] Capturar CSI real en mi habitación (yo moviéndome vs vacío)
- [ ] Entregable: dataset propio grabado y etiquetado

### Fase 2 — Presencia y movimiento (Semanas 3-5)
Objetivo: primer demo real.
- [x] Pipeline de preprocesado (limpiar fase, filtrar, normalizar)
      → src/preprocess.py: drop_null_subcarriers + hampel_filter + lowpass
      (Butterworth fase-cero), encadenados en preprocess_amplitude. Ventaneo en
      features.sliding_windows. Validado con sintético. Sesión 3.
- [ ] Detección presencia/movimiento en tiempo (casi) real
- [ ] Probar a través de una pared
- [ ] Entregable: demo en vivo que dice "hay alguien / no hay nadie"

### Fase 3 — Reconocimiento de actividad / caídas (Semanas 6-8)
Objetivo: subir un escalón, el que diferencia.
- [ ] Grabar dataset multi-actividad (andar, sentarse, caer, quieto)
- [ ] Modelo de clasificación de actividad (CNN o similar)
- [ ] Foco en detección de caídas (ángulo cuidado de mayores)
- [ ] Documentar honestamente dónde falla (generalización entre entornos)
- [ ] Entregable: clasificador de actividad con métricas

### Fase 4 — Publicar (Semanas 9-10)
Objetivo: convertir el trabajo en portfolio/tracción.
- [ ] Repo GitHub limpio con README decente
- [ ] Post técnico o vídeo del demo
- [ ] (Opcional) contactar con alguien que lo pueda usar
- [ ] Entregable: proyecto público y presentable

---

## 3. Lista de compra (rellenar en Fase 1)
- [x] 2× ESP32-WROOM-32 DevKitC USB-C ("tipo C") — AliExpress, 5,89 €/u = 11,78 €
      (incluye adaptador de bornes verde de regalo; no imprescindible)
- [ ] 2× cables USB-C de DATOS (no solo carga) — confirmar que la placa es USB-C
- [ ] (Opcional, NO comprar ahora) microSD — el CSI se guarda por USB en el PC
- [x] Driver serie: es CH340C → instalado CH341SER.EXE en Windows (OK).
      (No hace falta CP2102.)
Presupuesto real: ~12 € placas + cables

---

## 4. Decisiones tomadas
_(Aquí anotamos elecciones importantes y por qué, para no repensarlas.)_
- **CSI sintético primero, dataset real después.** Se valida el pipeline con
  una señal que controlamos (sabemos la verdad → si falla, el bug es nuestro).
  El valor duradero de la Fase 0 es el CÓDIGO (loader/limpieza/features/modelo),
  no el modelo entrenado (el CSI no generaliza entre entornos).
- **Dataset real objetivo: formato ESP32, no Intel 5300.** Así el loader y el
  preprocesado escritos ahora se reutilizan tal cual con nuestras capturas de
  la Fase 1 (~64 subportadoras, misma estructura de fila).
- **Features sobre AMPLITUD, no fase cruda.** La fase real trae CFO/SFO y
  necesita saneado; la amplitud es robusta y estándar para detección de mov.
- **Normalizar por nivel medio, NO z-score.** El z-score fuerza varianza
  unitaria y borra la fluctuación temporal, que es justo la señal.
- **Hardware: 2× ESP32-WROOM-32 (USB-C), no S3.** Aunque el S3 da mejor CSI,
  el WROOM-32 es el mejor soportado por ESP32-CSI-Tool y por los tutoriales/
  datasets → menos fricción para el primer montaje. Se usará el
  [ESP32-CSI-Tool de Steven Hernandez] (1 placa TX, 1 placa RX).

---

## 5. Registro de progreso
_(Al final de cada sesión: fecha + qué hice + qué me atascó + siguiente paso.)_

### 2026-07-06 — Sesión 1
- Hecho:
  - Entendido CSI vs RSSI y qué es una subportadora (64 "sensores" en frecuencia).
  - Montado entorno (.venv) con numpy/pandas/scipy/sklearn/matplotlib/jupyter.
  - Pipeline Fase 0 completo con CSI SINTÉTICO: cargar→visualizar→limpiar→
    features→clasificar→validar (notebooks/00_fase0_pipeline_csi.ipynb + src/).
  - Clasificador movimiento vs quieto ~100% a 25 dB; validado que se degrada
    a azar (~0.5) al bajar la SNR (no hay fuga de datos).
  - Decidido y PEDIDO hardware: 2× ESP32-WROOM-32 USB-C (AliExpress, ~10-14 €).
- Atascos:
  - Confusión al elegir la placa en AliExpress ("Expansion Board" vs la placa
    de desarrollo real). Aclarado: se necesita la "Placa de desarrollo
    ESP32-38PINS", NO la expansion/breakout board.
- Siguiente paso:
  - (Software, mientras llega el HW) Escribir el LOADER del formato CSV real del
    ESP32-CSI-Tool y conectarlo al pipeline existente. Cerrar Fase 0 con datos
    en formato real.

### 2026-07-09 — Sesión 2
- Hecho:
  - LOADER del formato real ESP32-CSI-Tool: src/load_esp32.py
    (load_esp32_csv → fila_a_amplitud → cargar_amplitudes). CSV real →
    matriz amplitud (n, 64). Validado contra example_csi.csv oficial del repo.
    Banco de pruebas en notebooks/01_loader_esp32.ipynb.
  - Confirmado que el pipeline existente (features.py) traga la matriz real
    sin cambios (np.abs de una amplitud = ella misma). Punto 1 cerrado.
  - Entorno de flasheo montado SIN hardware: ESP-IDF v4.3.6 (Offline Installer),
    repo ESP32-CSI-Tool clonado en Desktop\ESP32-CSI-Tool, driver CH340
    instalado, y active_ap compila limpio. Punto 2 cerrado.
- Atascos:
  - La extensión ESP-IDF de VSCode no mostraba el comando "Configure ESP-IDF
    Extension" y su instalador online ya no ofrece la v4.3. Solución: instalador
    OFFLINE oficial v4.3.6 desde dl.espressif.com. Sin fricción.
  - Jupyter cacheaba el módulo al editar el .py → resuelto con %autoreload 2.
  - os.chdir("..") frágil (subía dos niveles) → versión idempotente que solo
    sube si estás en notebooks/.
- Siguiente paso (día 14, al llegar las placas):
  - Flashear active_ap en placa A y active_sta en placa B (idf.py -p COMx flash).
  - Confirmar cuál imprime CSI_DATA y capturar: idf.py monitor | findstr "CSI_DATA" > captura.csv
  - Primera captura real: yo moviéndome vs habitación vacía (dataset etiquetado).
- Pendiente opcional sin HW: [HECHO] active_sta también compila; protocolo de
  captura escrito abajo (sección 7).

### 2026-07-13 — Sesión 3
- Hecho:
  - Módulo de preprocesado de amplitud real: src/preprocess.py.
    · drop_null_subcarriers: quita subportadoras nulas (std temporal ~0).
    · hampel_filter: anti-picos por mediana/MAD, quirúrgico (solo toca la
      muestra del pico, error 0 en el resto — validado).
    · lowpass: Butterworth paso-bajo con filtfilt (fase cero), deja la banda
      lenta ~0.2-3 Hz (movimiento/respiración) y mata el ruido rápido.
    · preprocess_amplitude: los encadena en el orden correcto
      (nulas → anti-picos → paso-bajo). Validado con las 3 suciedades juntas.
  - Ventaneo: features.sliding_windows (captura continua → ventanas solapadas
    (n_vent, win_len, n_sub); win_len=128 ≈1.28 s @100 Hz, hop=64). Validado.
  - Pipeline real montado end-to-end salvo enchufar el CSV:
    load → preprocess → sliding_windows → features → modelo.
  - DETECTOR DE RESPIRACIÓN (estrella polar) montado y validado en sintético:
    · synth_csi.generate_breathing: persona quieta, pecho = trayecto dinámico
      débil que oscila a f_b=bpm/60 (micro-desplazamiento, no cuerpo entero).
    · breathing.estimate_breathing: FFT temporal + zero-padding (afina el pico)
      + combinación de subportadoras (mean/weighted) + confianza (pico/media en
      banda). Clava BPM (error <0.2) y NO alucina en vacío (conf ~1.26).
  - CURVA DE ROBUSTEZ vs SNR (figures/robustez_respiracion.png, pilar de
    evaluación honesta). Hallazgo: la palanca que manda es la VENTANA LARGA,
    no la ponderación. 30 s → fiable ≥20 dB; 60 s → fiable ≥15 dB (ganancia de
    proceso). Ponderar no ayuda en sintético (subportadoras con SNR homogéneo);
    debería ayudar en real (SNR heterogéneo) → se deja puesta.
  - Regla operativa: ventana 60 s, apuntar a SNR ≥15 dB. El día 16 se mide el
    SNR real de la componente respiratoria: ≥20 dB → WROOM basta; <15-20 dB →
    exprimir software y, si sigue corto, front-end multi-antena (CSI ratio).
  - NOTEBOOK DE DEMO (wow packaging): notebooks/02_demo_respiracion.ipynb,
    ejecutado y con salidas guardadas. 4 paneles: waterfall crudo → PCA (SVD)
    que extrae la onda respiratoria limpia → espectro con pico a 14 rpm (conf
    9.5) → habitación vacía plana (conf 1.3, "no detecta"). Figura exportada a
    figures/demo_respiracion.png. Narrativa honesta: el crudo parece ruido, el
    DSP saca la respiración, y el sistema se niega cuando no hay señal.
  - Nota entorno: nbconvert roto (html5lib incompat. Python 3.13). Ejecutar
    notebooks con `jupyter execute` o nbclient, no con nbconvert.
  - BUILD DATASET real: src/dataset.py (build_dataset + etiqueta_de_nombre).
    Recorre data/raw/*.csv → cargar_amplitudes → hampel → lowpass → ventanea →
    etiqueta por nombre (clase_nn.csv). Mantiene las 64 subportadoras (no dropea
    nulas) para que las ventanas de archivos distintos se puedan apilar.
    Verificado con las 2 muestras renombradas: X (78,128,64), pipeline entero
    (build → features → RandomForest) corre end-to-end. Pipeline real CERRADO.
  - CAVEAT CLAVE para el día 16: la CV 5-fold sobre un solo archivo/clase INFLA
    el resultado (fuga: ventanas de la misma grabación comparten canal estático).
    Evaluación honesta = grabar VARIAS tandas por clase y testear cross-grabación
    (train en unas tandas, test en OTRAS). Por eso el protocolo pide 4-5 tandas.
  - SCAFFOLD EN TIEMPO REAL: src/stream.py, probado end-to-end con dry-run en
    vivo (un hilo escribe CSI al CSV mientras se lee).
    · CsiTailer: lectura incremental tipo 'tail -f' (offset en bytes, guarda
      linea parcial, salta corruptas). Verificado sin duplicados.
    · LiveMonitor: buffer deslizante + maquina de 3 estados. El discriminante
      movimiento/quieto NO es un umbral: es el CLASIFICADOR entrenado (features
      -> modelo), que se INYECTA. La respiracion se estima aparte con el
      estimador sobre el buffer largo. LECCION: mov vs quieto no se separa con
      un estadistico simple (a 25 dB el ruido domina), es clasificacion.
    · monitor_vivo: bucle que imprime el veredicto cada N s. Dia 16:
      monitor_vivo("data/raw/captura.csv", clf) en paralelo a la captura.
    · Quirks del generador sintetico (NO afectan a datos reales, solo a los
      tests): generate_window define el movimiento en tiempo normalizado por
      ventana (no en Hz), y concatenar ventanas crea saltos de canal falsos.
      Cada estado se prueba con la generacion que le toca.
    · umbral_resp=2.0 y el clf son PLACEHOLDER: calibrar/entrenar con datos
      reales el dia 16.
  - PENDIENTE TÉCNICO: vectorizar hampel_filter (ahora es bucle Python; en vivo
    con capturas de 6000+ muestras será lento). Para el día 16.
- Siguiente paso (día 16, con placas — todo el software ya está listo):
  1. Flashear (sección 7) y grabar VARIAS tandas de vacio/ y mov/ a data/raw/
     (nombres clase_nn.csv), para poder evaluar cross-grabación sin fuga.
  2. build_dataset("data/raw") → extract_features → entrenar clf; evaluar
     train-en-unas-tandas / test-en-OTRAS (no CV aleatoria).
  3. Grabar 60 s de persona quieta respirando → estimate_breathing sobre datos
     REALES; medir el SNR de la componente respiratoria (¿≥15-20 dB?).
  4. Inyectar el clf real en monitor_vivo y hacer el demo en vivo real; calibrar
     umbral_resp con las capturas.
  5. Decidir hardware según el SNR medido: WROOM basta / hace falta multi-antena.
- Packaging (Fase 4) HECHO en parte:
  - Repo git propio inicializado en Desktop/Sensing (rama main, .gitignore
    correcto). PUBLICADO en GitHub: https://github.com/marcrubii/pulse-wifi-sensing
  - README reescrito con demo (figuras), diagrama del pipeline, tabla de módulos,
    estado y sección de honestidad (límite de generalización entre salas).
  - TODO EL REPO EN INGLÉS: README, los 8 módulos de src/ (docstrings/comentarios),
    estados del monitor en vivo, notebook (texto/comentarios/etiquetas de gráficos)
    y las 2 figuras regeneradas. Identificadores en español se mantienen (para no
    romper imports entre módulos). La bitácora se queda en español (log personal).
  - Pendiente: writeup técnico de 1-2 páginas posicionando vs literatura
    (802.11bf; línea de sensing/vital-signs por CSI) — mejor hacerlo con una
    MÉTRICA REAL que contar, o sea después del día 16.
- Estrategia (importante):
  - Estrella polar fijada: detección de RESPIRACIÓN de persona quieta a través
    de pared (micro-movimiento del pecho ~0.2-0.4 Hz). Presencia/movimiento en
    vivo como base debajo.
  - Objetivo: versión lo más top posible, no "hecho y ya". Lo que separa
    fascinante de mediocre: (1) un demo con momento wow (respiración/atravesar
    pared), (2) evaluación honesta cruzando salas (entrenar en una, testear en
    otra; sin fuga por split aleatorio).
  - Presupuesto NO es límite: comprar mejor front-end si mejora calidad (más
    antenas RX para CSI ratio, mejor chipset, GPU cloud para modelos mayores).
- Nota: las placas llegan ahora el 16/07 (antes 14/07). El "día 14" de la
  sección 7 pasa a ser el día 16.
- Siguiente paso (día 16): flashear, capturar vacio/mov, y escribir el
  "build dataset desde data/raw/" (load+preprocess+ventaneo+etiqueta por nombre)
  para enchufar el CSV real al pipeline y sacar el primer modelo real.

---

## 6. Preguntas abiertas para Claude
_(Cosas que quiero resolver la próxima vez.)_
- (pendiente)

---

## 7. Protocolo de captura (guion para el día 14)

**Posicionamiento estratégico (visión startup):** foco en UN vertical —
presencia/caídas para cuidado de mayores SIN cámara (privacidad). La ambición
está en clavar ese caso y contarlo bien, no en sensar toda la casa. El modelo es
pequeño (entrena en segundos/minutos en el portátil) → reentrenable en el sitio.
Recordar: el CSI NO generaliza entre salas → para demo en sitio nuevo se
CALIBRA in situ (grabar 2 min por clase + reentrenar). Venderlo como feature:
"se adapta a cualquier sala en ~10 min".

**Montaje físico:**
- Habitación: dormitorio de Marc (un solo enlace, una sala).
- 2 placas FIJAS (cinta), separadas ~2-3 m, misma habitación. Una con active_ap
  (receptor, el que loguea CSI) y otra con active_sta. Que NO se muevan entre
  capturas (si se mueven las placas, se ensucia la señal).

**Clases — PRIMERA toma (empezar con 2, no con caídas):**
- `vacio` → habitación sin nadie (o Marc totalmente quieto).
- `mov`   → Marc andando entre las placas.
- (Fase 3, más adelante: añadir sentarse/caer. Caídas = lo más difícil de
  capturar bien: evento <1 s, pocos ejemplos, desbalanceado, necesita colchoneta.)

**Cuánto grabar (balanceado):**
- 4-5 tandas de ~1 min por clase. A ~100 pkt/s, 1 min ≈ 6000 filas → cientos de
  ventanas. Mismo tiempo de `vacio` que de `mov`.

**Captura a CSV (Windows, terminal ESP-IDF, monitorizando la placa active_ap):**
```
idf.py monitor | findstr "CSI_DATA" > data/raw/<clase>_<nn>.csv
```
Etiquetado POR NOMBRE DE ARCHIVO: data/raw/vacio_01.csv, mov_01.csv, ...
La clase se lee luego del nombre. Simple.

**REGLA DE ORO:** entre `vacio` y `mov` cambiar UNA sola cosa (que haya
movimiento o no). Misma posición de placas, misma sala, misma hora. Así, si el
modelo distingue, es por el movimiento y no por un artefacto.

**Checklist día 14:**
1. Enchufar placas → Admin. de dispositivos → anotar COMx de cada una.
2. Flashear: active_ap en placa A, active_sta en placa B (idf.py -p COMx flash).
3. Comprobar que active_ap imprime líneas CSI_DATA (idf.py monitor).
4. Grabar las tandas de `vacio` y `mov` a data/raw/ con el comando de arriba.
5. Cargar con src/load_esp32.py y meter al pipeline → primer modelo real.