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

**TECHO DE AMBICIÓN (acordado en Sesión 5 — Marc quiere llegar al máximo).**
La escalera, de lo hecho a lo imposible, con el porqué técnico de cada peldaño:
1. **Presencia / movimiento en vivo** — la base. Datos ya grabados (Sesión 5).
   No es el "wow": es lo que hace creíble al wow.
2. **Evaluación cross-sala** (entrenar en una, testear en otra) — NO es ambición,
   es **credibilidad**. Sin ese número, cualquiera que sepa de CSI asume fuga
   ante un 98%, y casi siempre acierta. Convierte "mi demo funciona" en "lo he
   demostrado". Barato y va de propina.
3. **Respiración de persona quieta** — LA ESTRELLA POLAR. Micro-movimiento del
   pecho ~0.2-0.4 Hz. Documentado en la literatura con hardware como el nuestro.
4. **Respiración a través de una pared** — el momento wow. Misma técnica, menos
   SNR.
5. **Espectrogramas Doppler + CNN pequeña** para actividades (andar/sentarse/
   caer). Los 100 Hz reales dan margen de sobra.
6. **Frecuencia cardíaca** — el techo real. Movimiento ~10x menor que la
   respiración. Aquí el front-end multi-antena (CSI ratio) deja de ser capricho
   y pasa a ser necesario.
**IMPOSIBLE con este hardware (no insistir):** representar la figura/silueta a
través de la pared, tipo RF-Pose / RF-Capture del MIT. Ellos usan un radar FMCW
propio con un ARRAY de decenas de antenas: forman haces y barren el espacio. Con
1 TX y 1 RX de una antena cada uno **no hay resolución espacial** — no se puede
formar imagen, igual que no se hace una foto con un solo píxel. No es cuestión
de software ni de esfuerzo: falta la apertura física.

**Por qué el ESP32 tiene techo:** tiene UNA sola cadena de RX (ni el S3, ni los
módulos con "diversidad de antena", que conmutan pero reciben por una a la vez).
Eso descarta la **CSI ratio** — dividir el CSI de dos antenas del mismo receptor
para que el ruido de fase (desajustes de reloj TX/RX, enormes en hardware
comercial) se cancele por ser común a ambas. Sin CSI ratio, la fase es
inutilizable y solo queda la amplitud. Upgrade si el SNR respiratorio se queda
corto: **Intel AX200/AX210 + PicoScenes** (moderno, hasta 160 MHz, varias
antenas, Linux actual) > Intel 5300 + tool de Halperin (3 antenas, el clásico de
los papers de respiración, pero kernel viejo = mucha fricción) > Nexmon en
Raspberry Pi 4 (256 subportadoras a 80 MHz, pero 1 antena → da resolución, no
CSI ratio). Decidir CON el SNR medido delante, no antes.

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
  - Pulido final del repo: LICENSE MIT añadido; scratch_tr.py (borrador) eliminado;
    notebooks 00/01 de Fase 0 sacados del repo (siguen en local, gitignored) → el
    repo publica solo el notebook de demo 02. Raíz limpia. Todo sincronizado.
  - Pendiente Fase 4 (para después del día 16, con métrica real): writeup técnico
    1-2 págs posicionando vs literatura. Y opcional en la web: añadir topics.
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

### 2026-07-14 — Sesión 4 (HARDWARE VIVO, llegó 2 días antes)
- Placas flasheadas y drivers CP210x OK. **COM7 = AP** (active_ap, el que loguea
  CSI; crea la red) · **COM3 = sta** (active_sta, se conecta y emite paquetes).
  Los mensajes de "connect to ap / wifi not connected" son de la STA, no del AP.
  Marcar las placas con cinta: son idénticas por fuera y se confunden.
- Tool usado: **ESP32-CSI-Tool** en Desktop/ESP32-CSI-Tool (no esp-csi).
- TRES bugs de puesta en marcha, ninguno obvio. Documentados por si se repite:
  1. **AP en boot loop**: `esp_wifi_set_csi(1)` devolvía ESP_FAIL en
     csi_component.h:96 → ESP_ERROR_CHECK aborta → reinicia sin parar. Causa:
     `CONFIG_ESP32_WIFI_CSI_ENABLED` sin poner (es `default n` en ESP-IDF y el
     tool NO trae sdkconfig.defaults que lo active). Arreglo: menuconfig →
     Component config → Wi-Fi → "WiFi CSI(Channel State Information)".
     · LECCIÓN: son DOS interruptores distintos. `SHOULD_COLLECT_CSI` (del tool)
       puede decir 1 en la cabecera y aun así faltar el de ESP-IDF, que es el
       que compila el CSI en el driver. Hacen falta los dos.
  2. **Tasa real 12 Hz en vez de 100**: `CONFIG_FREERTOS_HZ=100` (default) →
     1 tick = 10 ms → el `vTaskDelay(w)` de sockets_component.h:65 duerme 100 ms
     en vez de 10. El tool asume tick = 1000 Hz. La sta emitía a 10 pkt/s.
     Arreglo en la **STA**: menuconfig → Component config → FreeRTOS →
     "Tick rate (Hz)" = **1000**. Entonces PACKET_RATE=100 se cumple de verdad.
  3. **Serie a 115200 = techo ~21 líneas/s** (cada fila CSI son ~540 bytes) →
     habría capado los 100 Hz. Arreglo en el **AP**: Component config →
     **Common ESP-related** → "Channel for console output" = **Custom UART**
     (¡el baudrate está OCULTO hasta elegir Custom!) → "UART console baud rate"
     = **921600**; y Serial flasher config → "'idf.py monitor' baud rate" =
     "Same as UART console baud rate" (si no, ves basura ilegible).
     · No tocar los pines de Custom UART: UART0 + GPIO 1/3 ya son los correctos.
  - **Resultado: 98.6 filas/s estables** (de 12 → 100 Hz). Hacían falta los dos
    cambios: el tick para que emita, el baudrate para poder contarlo.
- **src/capture.py** (NUEVO): lee stdin de `idf.py monitor`, descarta el
  calentamiento y graba N segundos de líneas CSI_DATA a CSV.
  `idf.py -p COM7 monitor | python src/capture.py <out.csv> <segundos> <calent.>`
  · El calentamiento es IMPRESCINDIBLE: `idf.py monitor` resetea el AP, y los
    primeros ~4 s van a 1-13 pkt/s mientras la sta reconecta. Ese hueco de
    muestreo irregular al principio ensucia el ventaneo (que asume tasa uniforme).
  · Mejor que el `findstr` de la sección 7: controla duración y descarta el
    arranque. El findstr funciona pero no hace ni lo uno ni lo otro.
- **Validación con datos REALES** (data/raw/prueba.csv, 20 s):
  · `cargar_amplitudes` → (1974, 64) float64. El pipeline traga el CSV real
    SIN tocar una línea de código. load_esp32.py estaba bien.
  · Muestreo: mediana 10.1 ms, p95 16.2 ms, peor hueco 75.8 ms, **0 huecos
    >100 ms**. Uniforme.
  · Tramas homogéneas: las 1974 filas son sig_mode=1 / len=384. La mezcla de
    128/384 que se ve al arrancar es solo de las tramas de asociación y DHCP;
    en régimen estable solo hay UDP de la sta. NO hay que filtrar por sig_mode.
  · OJO al despiste: la columna `len` dice 384, pero el corchete SIEMPRE trae
    128 números → 64 subportadoras. csi_component.h:54 trunca a la parte LLTF
    cuando SHOULD_COLLECT_ONLY_LLTF=1. `len` es el buffer completo del chip.
- Ganancia estratégica: 100 Hz vs los 12 iniciales = ~8x muestras por ventana
  ≈ **+9 dB de ganancia de proceso** en el espectro. Relevante para la medida
  del SNR respiratorio que decide el hardware (WROOM vs multi-antena).
- Truco de montaje: la sta **no necesita datos, solo corriente** → cargador de
  móvil o batería externa. Así se separan 2-3 m sin depender de cables USB al
  portátil. Solo el AP necesita el USB (es quien manda el CSI).
- Siguiente paso (mañana): grabar las tandas y sacar el primer modelo real.

### 2026-07-15 — Sesión 5 (FASE 2 CERRADA: dataset real + modelo + demo en vivo)
- **RESULTADO: 0.987 ± 0.012 de accuracy con evaluación CROSS-GRABACIÓN** (5 folds,
  cada uno entrena con 8 tandas y testea con 2 que el modelo no vio nunca).
  Y **`monitor_vivo` funcionando en tiempo real** con el clf real inyectado:
  canta MOTION al andar y EMPTY al pararse. Fase 2 cerrada.
- Contexto honesto de ese 98.7%: es UNA habitación, UNA geometría, UNA persona,
  15 minutos. "Sala vacía vs tío andando" es la discriminación más fácil que
  existe (12x de energía en la banda 0.5-5 Hz). **Es el suelo, no el logro.**
  El número que importa es el de la segunda sala. Ver TECHO DE AMBICIÓN (§1).
- **Comparación honesta vs fuga: NO concluyente, y hay que decirlo.** CV aleatoria
  0.991 vs cross-grabación 0.987. La fuga de la CV aleatoria es real (las ventanas
  solapan al 50% → ventanas casi idénticas caen a los dos lados del split), pero
  con los dos números pegados al techo no aflora. Ese experimento valdrá cuando
  haya clases difíciles (persona quieta vs vacío, caídas).
- **Las features de nivel absoluto NO se están usando**: quitar `amp_mean` y
  `amp_std` da 0.984 vs 0.987. El modelo va por variabilidad temporal, no por el
  nivel de señal. Buena noticia para transferir a otra sala: la variabilidad
  viaja, el nivel absoluto no.
- DATASET: 10 tandas (5 `vacio` + 5 `mov`, 60 s c/u) → **986 ventanas
  (986, 128, 64)**, balanceado. `data/raw` contiene SOLO las 10 tandas.
  · `data/synthetic/` (NUEVO): ahí se movieron prueba.csv y los 2 sample_*.csv
    sintéticos, que si no `build_dataset` los cogía como clases falsas.
  · `data/live/` (NUEVO): capturas del monitor en vivo, fuera del dataset.
- **HALLAZGO GRANDE — la adaptación de tasa chivaba la clase.** Está documentado
  en detalle en §7. Resumen: el cuerpo de Marc mejora el enlace ~3 dB → menos
  caída a tramas legacy → la composición de tramas (y la tasa útil tras filtrar:
  87 Hz en mov vs 39 Hz en vacío) delataba la presencia. Arreglado fijando la
  tasa PHY en la sta (`esp_wifi_internal_set_fix_rate`, MCS0_LGI, con AMPDU TX
  desactivado) → **legacy 0.00%, 100% de filas útiles, tasas 101-111 Hz sin sesgo
  por clase**. Regalo inesperado: la desviación del RSSI cayó de 3.4 a **0.6 dB**
  (todas las tramas pasan por la misma cadena) → el suelo de ruido instrumental
  se dividió por 6 en un día.
- Interferencia: el router de casa (MOVISTAR, 2.4 GHz) está en el **canal 7 al
  91%**, y el ESP32 estaba en el 6 → solape casi total. Cambiado a **canal 1**
  (libre; el 7 ocupa 2434-2456 MHz, el 1 va de 2401-2423). Efecto medido: tramas
  legacy 14.9% → 5.8%, y el enlace mucho más estable (RSSI mín. −86 → −77).
- `src/capture.py` REESCRITO: ahora abre el puerto serie él mismo con pyserial en
  vez de leer de `idf.py monitor`. Motivo: `idf_monitor.py:393` hace
  `self.serial.rts = True  # Force an RTS reset on open` — **resetea el AP en cada
  captura**, la sta se desconecta y hay que esperar el reenganche. Sin flag para
  evitarlo en IDF 4.3. Abriendo el puerto con `dtr=False`/`rts=False` ANTES del
  `open()`, el AP ni se entera. Uso: `python src/capture.py COM3 <out.csv> <s> <calent.>`
  · Ya no hace falta la terminal ESP-IDF: vale cualquier cmd. Pero pyserial tiene
    que estar en el Python del sistema (`pip install pyserial`), no solo en el de
    Espressif.
- `src/load_esp32.py`: `load_esp32_csv(path, sig_mode=1)` filtra por tipo de trama.
  A igual RSSI, legacy y HT salen del chip en escalas de amplitud distintas
  (~19 vs ~13) → mezclarlas metía saltos del 50% que parecían movimiento (38% de
  la "actividad" en sala quieta era eso). Con la tasa fija ya no hay legacy, pero
  el filtro se queda como red de seguridad.
- `src/dataset.py`: `build_dataset` ahora devuelve **`grupos`** (la tanda de
  origen de cada ventana) — sin eso no hay evaluación cross-grabación posible.
  Y `fs` por defecto **107.0**, la tasa real medida (era 100.0).
  · **OJO para la respiración:** si en la FFT usas 100 cuando la realidad son 107,
    todas las frecuencias salen 7% desplazadas → 15 rpm se leerían como 14.
- **DOS BUGS en `src/stream.py` que habrían roto el demo en vivo:**
  1. `mov_label=1` por defecto, pero `clases = ['mov','vacio']` ordenado
     alfabéticamente → `mov` es el **0**. El monitor habría dicho MOTION con la
     sala vacía. `monitor_vivo` ahora expone `mov_label`; pasar `clases.index("mov")`.
  2. **Training/serving skew (el sutil):** `veredicto()` llamaba a
     `window_features()` sobre el buffer CRUDO, pero `build_dataset` aplica
     `hampel_filter` + `lowpass(fc=5)` ANTES de ventanear. Modelo entrenado con
     datos filtrados, servido con datos sin filtrar. No peta: solo funciona mal.
     Arreglado aplicando la misma cadena en `veredicto()`.
- Nota de método (repetida 3 veces hoy): al pegar bloques de reemplazo se perdió
  el resto de la función (en `load_esp32.py`, en `veredicto` y en `monitor_vivo`,
  que se quedó sin bucle y por eso no imprimía nada). Verificar siempre lo pegado.
- `local_timestamp` (col 18) es un contador de **32 bits en µs → se desborda cada
  71.6 min**. Pasó en mov_04 (fila 2563: 4294963303 → 13631). Los datos están
  bien; lo que falla es cualquier análisis que reste timestamps sin tratar el
  wrap. La col 23 (`real_timestamp`, en segundos) no se desborda.
- Deriva observada: el RSSI de los `vacio` cayó monótonamente −73.7 → −77.0 en
  los 15 min de sesión, sin tocar nada. Algo se mueve despacio (cinta cediendo,
  temperatura). Es un argumento a favor de la evaluación cross-sala: si el canal
  deriva 3 dB en un cuarto de hora, un modelo atado a un montaje es frágil.
- **PENDIENTE (aviso):** el clf está atado a ESTA geometría. Si se despegan las
  placas, deja de servir en vivo y hay que regrabar las 10 tandas.
- Siguiente sesión: bajar la sta a la altura del pecho tumbado (§7 punto 5) e ir
  a por la RESPIRACIÓN. Y después, la segunda sala.

### 2026-07-15 (tarde) — Sesión 5b: **RESPIRACIÓN REAL DETECTADA**
- **HITO: la estrella polar funciona.** Respiración pautada a **6 rpm** → el
  sistema mide **0.100 Hz = 6.0 rpm exactos, con SNR de 39.1 dB**. Con dos placas
  de 8 €. Ficheros en `data/live/`.
- **Geometría de respiración** (cruzando la cama, §7): sta y AP a los lados de la
  cama a la altura del pecho tumbado (~65 cm), persona boca arriba en medio, el
  enlace le cruza el esternón. **RSSI −61.7 dBm: 8 dB mejor** que la geometría de
  vacio/mov. Sin mesita: una placa pegada a la pared, la otra sobre libros.
- **PROTOCOLO DE MEDIDA (esto es lo que hace que valga):**
  1. **Verdad de referencia obligatoria.** Respirar a ritmo PAUTADO con reloj.
     6 rpm = inspirar 5 s / espirar 5 s = 0.100 Hz. Sin ground truth, un pico
     bonito no demuestra nada.
  2. **Control negativo obligatorio.** Misma captura con la sala vacía. Si el
     estimador también "encuentra" respiración sin nadie, todo lo demás da igual.
  3. **Medir la potencia EN la frecuencia esperada, no el mejor pico de la banda.**
     Si buscas el máximo, siempre encuentras uno — también en el vacío.
  4. **Predicción falsable.** Cambiar el ritmo pautado y comprobar que el pico
     SIGUE. Es lo que separa "me sale un pico" de "he medido algo".
- **El experimento que lo demostró:** pautado 12 rpm → midió 9.0 rpm (¡el sistema
  tenía razón, el cronómetro mental de Marc estaba mal — contar segundos de
  cabeza va lento!). Repetido a 6 rpm → **0.100 Hz clavado**. El pico siguió al
  ritmo exactamente como se predijo. Control de vacío: 7.9 dB en frecuencia
  aleatoria = ruido.
- **DECISIÓN DE HARDWARE (la que la bitácora dejó pendiente): el WROOM BASTA.**
  El listón era ≥20 dB de SNR respiratorio. Medido: **39.1 dB en línea de visión**,
  19 dB de margen. Para respiración con LOS no hay que comprar nada.
- **A través de la puerta cerrada (madera, cuesta solo 1.4 dB — NO es un tabique;
  no decir "a través de una pared"):**
  ```
                            RSSI     SNR@0.1Hz (mejor subportadora)
  sta DENTRO (LOS)         -66.7  ->  42.2 dB   solido
  FUERA, puerta ABIERTA    -67.8  ->  24.6 dB   detecta
  FUERA, CERRADA (1a)      -62.6  ->   3.3 dB   nada
  FUERA, CERRADA (replica) -70.8  ->  17.3 dB   detecta
  VACIO (control)          -54.7  ->   8.1 dB
  ```
  **Estado honesto: sin línea de visión DETECTA, pero es frágil.** Dos medidas de
  la misma condición dan 3.3 y 17.3 dB — 14 dB de diferencia. Y el RSSI se movió
  8 dB entre ellas, o sea que **no eran la misma condición**: cambió algo (postura,
  ángulo de la puerta, un brazo). Al perder el camino directo, la sensibilidad a
  la geometría exacta se dispara.
- **LECCIÓN DE MÉTODO, LA MÁS IMPORTANTE DEL DÍA:** con la primera medida de
  puerta cerrada (3.3 dB), Claude construyó una teoría entera y elegante ("sin
  camino directo la modulación se diluye; el RSSI no mide la capacidad de
  sensar") que encajaba con la física y **era falsa**. Estaba montada sobre UNA
  medida mala. **Marc pidió repetirla y la tumbó.** n=1 no es un resultado, por
  muy bien que suene la explicación. Replicar antes de concluir — sobre todo
  cuando el resultado te gusta.
- **PENDIENTE — lo que falta para poder afirmar algo:**
  · **5 repeticiones de puerta CERRADA + 5 de puerta ABIERTA**, y comparar
    distribuciones, no puntos. Con esta varianza, un dato suelto no vale.
  · **Control de vacío para las condiciones de puerta.** El único control que hay
    se grabó con la sta DENTRO. Hace falta sala vacía + sta fuera + puerta
    cerrada, o no hay con qué comparar.
  · **El análisis de hoy fue un script de usar y tirar** (SVD + FFT + SNR en la
    frecuencia diana). NO está en el repo. Hay que llevarlo a `breathing.py` /
    un notebook: cargar, SVD, buscar potencia en la frecuencia pautada, SNR
    contra la mediana de 1-5 Hz, barrer componentes SVD y subportadoras.
  · Calibrar `umbral_resp` de `stream.py` (sigue en 2.0, placeholder) con estos
    datos reales: en vivo daba `breathing 10 bpm` con conf 2.1 sobre una sala
    donde no había nadie respirando pautado.
- Por qué esto sigue justificando **más receptores** (y ahora con datos): no
  porque sin LOS sea imposible, sino porque **con un solo enlace es una lotería**.
  4 receptores pasivos = 4 tiradas independientes, y te quedas con el que ese día
  tenga un camino que cruce el pecho. Ver TECHO DE AMBICIÓN (§1).

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
- Montaje concreto (Sesión 5): **sta pegada a la pared sobre el cabecero de la
  cama a ~1.3 m** (con cargador de móvil) · **AP en el borde de la mesa a ~90 cm**
  (levantado sobre una caja, mirando a la habitación). Las dos **DE PIE y con la
  serpentina del PCB hacia arriba** — la antena es el zigzag de cobre del extremo
  opuesto al USB, no los pines. Que las dos coincidan en orientación importa más
  que la altura exacta.
- **NO apoyar las placas sobre metal.** Primer intento: la sta encima del
  radiador → plancha metálica pegada a la antena, la desintoniza y hace de
  espejo. Es el peor sitio de la habitación.
- Objetivo de la geometría: que la línea imaginaria entre las 2 placas cruce la
  sala a la altura del pecho y se pueda atravesar andando.

**TRAMPA DE ALIMENTACIÓN (Sesión 5, costó ~30 min):** la sta con cargador de
móvil no conectaba (el AP levantaba el softAP, imprimía la cabecera CSV y nunca
salía el `DHCP server assigned IP to a station`). **El LED rojo estaba encendido
y engañaba**: solo indica que llega algo al regulador, no que la placa funcione.
Causa: **el cable USB, no el cargador**. Un cable fino de carga tiene demasiada
resistencia; el ESP32 consume ~100 mA en reposo pero pega picos de 300-400 mA al
transmitir WiFi → cae la tensión → brownout → reset en bucle.
- Test decisivo (30 s, aísla la variable sin leer el log de la sta): enchufar la
  sta al USB del portátil y mirar la ventana del AP. Si entra → es la
  alimentación. Luego, con el MISMO cable que funciona, probar el cargador →
  distingue si el culpable es el cable o el cargador.
- Regla: usar cable grueso y corto. Plan B si falla el cargador: cable USB largo
  (2-3 m) desde el portátil; el cable es estático y no afecta a las medidas.

**Clases — PRIMERA toma (empezar con 2, no con caídas):**
- `vacio` → habitación sin nadie (o Marc totalmente quieto).
- `mov`   → Marc andando entre las placas.
- (Fase 3, más adelante: añadir sentarse/caer. Caídas = lo más difícil de
  capturar bien: evento <1 s, pocos ejemplos, desbalanceado, necesita colchoneta.)

**Cuánto grabar (balanceado):**
- 4-5 tandas de ~1 min por clase. A ~100 pkt/s, 1 min ≈ 6000 filas → cientos de
  ventanas. Mismo tiempo de `vacio` que de `mov`.

**Captura a CSV (Windows, terminal ESP-IDF, desde la carpeta active_ap):**
_(Sesión 4: el `findstr` original funciona pero no controla duración ni descarta
el calentamiento. Usar src/capture.py, ver Sesión 4.)_
```
cd C:\Users\Marc\Desktop\ESP32-CSI-Tool\active_ap

REM vacío — calentamiento 25 s para dar tiempo a salir y cerrar la puerta
idf.py -p COM7 monitor | python C:\Users\Marc\Desktop\Sensing\src\capture.py C:\Users\Marc\Desktop\Sensing\data\raw\vacio_01.csv 60 25

REM movimiento — andar entre las placas desde el lanzamiento hasta el LISTO
idf.py -p COM7 monitor | python C:\Users\Marc\Desktop\Sensing\src\capture.py C:\Users\Marc\Desktop\Sensing\data\raw\mov_01.csv 60 8
```
Etiquetado POR NOMBRE DE ARCHIVO: data/raw/vacio_01.csv, mov_01.csv, ...
La clase se lee luego del nombre. Simple.
A ~100 Hz, 60 s ≈ 6000 filas por tanda.

**ALTERNAR las clases** (vacio_01, mov_01, vacio_02, mov_02, ...) en vez de
grabar las 5 seguidas de cada una: si todo el vacío se graba a las 22:30 y todo
el movimiento a las 22:50, cualquier deriva de la sala en esos 20 min se cuela
como pista falsa y el modelo acierta por el motivo equivocado. Alternando, esa
deriva se reparte entre las dos clases.

**REGLA DE ORO:** entre `vacio` y `mov` cambiar UNA sola cosa (que haya
movimiento o no). Misma posición de placas, misma sala, misma hora. Así, si el
modelo distingue, es por el movimiento y no por un artefacto.

**Confusores concretos detectados en Sesión 5 (aplicar en las 10 tandas):**
- **Ventilador de techo APAGADO.** Aspas girando = movimiento real y continuo →
  contamina `vacio` (la sala "vacía" parece tener movimiento) y arrasaría la
  respiración. Se vio en los datos: la actividad subió de 2.031 a 2.404 con el
  ventilador puesto. Lo mismo aplica a cortinas con la ventana abierta.
- **Móvil en MODO AVIÓN y en un sitio fijo de la mesa, también en `vacio`.**
  Error cometido: llevárselo al salir → el móvil estaba presente SOLO en `mov`,
  perfectamente correlacionado con la clase.
- **Puerta cerrada en las DOS clases** (es un reflector grande).
- Portátil, torre, silla y muebles: da igual dónde estén, pero IDÉNTICOS en las
  10 tandas. Lo que varía al azar en todas por igual es ruido (inofensivo); lo
  que cambia sistemáticamente entre clases es pista falsa (fatal).

**FUGA GRAVE: la adaptación de tasa chiva la clase (Sesión 5).**
Medido en las 3 primeras tandas: `mov_01` 12.0% de tramas legacy, `vacio_01`
36.5%, `vacio_02` 58.5%. Causa: con Marc en la habitación el enlace es ~3 dB más
fuerte (RSSI HT −71.9 vs −75.0) porque su cuerpo añade camino de reflexión; con
mejor enlace, la adaptación de tasa mantiene HT, y con la sala vacía se cae a
legacy. Consecuencias:
1. La composición de tramas es un proxy directo de "hay alguien" → atajo que no
   generaliza a otra sala/persona.
2. Peor: tras filtrar a sig_mode=1, `mov` queda a ~87 Hz y `vacio_02` a ~39 Hz →
   una ventana de 128 muestras abarca 1.5 s en `mov` y 3.3 s en `vacio`. Las
   ventanas dejan de medir lo mismo → fuga directa.
**Arreglo (en la STA): fijar la tasa PHY.**
- `#include "esp_private/wifi.h"` (tiene extern "C", va bien desde main.cc).
- En `app_main`, tras `station_init()`:
  `ESP_ERROR_CHECK(esp_wifi_internal_set_fix_rate(WIFI_IF_STA, true, WIFI_PHY_RATE_MCS0_LGI));`
- menuconfig de la STA → Component config → Wi-Fi → **desmarcar `WiFi AMPDU TX`**.
  La API devuelve ESP_ERR_NOT_SUPPORTED si el AMPDU TX está activo → boot loop.
- MCS0_LGI = la tasa HT más robusta (6.5 Mbps); a −75 dBm sobra.
- Verificación: `legacy ~0%` y misma tasa útil con y sin persona en la sala.
**LECCIÓN DE MÉTODO:** esto se cazó mirando los datos tanda a tanda, con 3
grabadas. Con las 10 hechas y un 98% de accuracy, se habría creído — y el modelo
habría estado leyendo el % de tramas legacy, no el movimiento. Analizar SIEMPRE
las primeras tandas antes de grabar el resto.

**Checklist día 14:** — HECHO en Sesión 4 (puntos 1-3). Placas vivas a 98.6 Hz,
pipeline validado con datos reales. Queda el punto 4 en adelante.

**Checklist para la próxima sesión (grabar y primer modelo real):**
1. Montaje: placas FIJAS con cinta a 2-3 m, misma sala. La sta con cargador de
   móvil (no necesita datos), el AP al USB del portátil. No moverlas en toda la
   sesión.
2. Grabar 5 tandas `vacio_NN.csv` + 5 `mov_NN.csv` (60 s cada una), ALTERNANDO
   clases, con los comandos de arriba. ~15 min. Borrar data/raw/prueba.csv
   (y las 2 sample_*.csv sintéticas, que no deben mezclarse con el dataset real).
3. `build_dataset("data/raw")` → `extract_features` → entrenar clf.
4. Evaluación HONESTA: train en unas tandas / test en OTRAS (cross-grabación).
   NO usar CV aleatoria: las ventanas de una misma grabación comparten canal
   estático y el resultado se infla (ver caveat de Sesión 3).
5. Grabar 60 s de persona quieta respirando → `estimate_breathing` sobre datos
   REALES → medir el SNR de la componente respiratoria (¿≥15-20 dB?).
   · **ANTES de esto hay que BAJAR la sta.** Montaje de Sesión 5: sta en la
     pared a ~1.3 m, AP en la mesa a ~90 cm → la línea del enlace va de 90 a
     130 cm, pero el pecho tumbado en la cama queda a ~60 cm del suelo. O sea
     el enlace pasa ~50 cm POR ENCIMA del pecho, y a 2.4 GHz la 1ª zona de
     Fresnel tiene ~30 cm de radio (λ=12.5 cm, enlace ~3 m) → el pecho queda
     FUERA de la zona sensible. Para respiración: bajar la sta a la altura del
     pecho tumbado. Requiere regrabar, así que no afecta a las tandas de
     vacio/mov (que son de pie y sí cortan el enlace).
   · Compromiso conocido: la geometría buena para respiración (línea sobre la
     cama, a la altura del pecho tumbado) NO es la mejor para "andar entre las
     placas". No hay óptimo para las dos a la vez. Priorizar respiración
     (estrella polar) y para `mov` cruzar la línea por los pies de la cama.
   · **Subir el RSSI antes de la sesión de respiración.** Montaje de Sesión 5
     mide **−70 dBm** a 3 m; en espacio libre tocarían ~−30, o sea faltan ~40 dB
     y NO es la distancia. Sospechas: (1) la sta está pegada PLANA contra la
     pared → el yeso justo detrás de la antena absorbe; (2) el AP está metido
     entre el monitor, los altavoces y el teclado. Arreglo: calzar la sta 2-3 cm
     separada de la pared (corcho/caja) y sacar el AP del nido de metal. Para
     vacio/mov da igual (un cuerpo cruzando el enlace es señal enorme y −70 con
     suelo de ruido −96 = 26 dB, por encima del objetivo), pero para las
     micro-modulaciones del pecho cada dB cuenta.
6. Inyectar el clf real en `monitor_vivo` + calibrar `umbral_resp` (ahora es un
   placeholder = 2.0). Demo en vivo real.
7. Decidir hardware según el SNR medido: WROOM basta / hace falta multi-antena.
- Pendiente técnico de Sesión 3, ahora más urgente: **vectorizar hampel_filter**
  (es un bucle Python; con 6000 filas por tanda × 10 tandas se va a notar).