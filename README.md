# Segmentación de audio

Proyecto final de Muestreo y Procesamiento de Señales: dada una grabación, dividirla automáticamente en intervalos de **música**, **voz**, **ruido** y **silencio**.

## Consigna

Dada una señal de audio, segmentarla automáticamente en categorías, como pueden ser música, voz hablada, silencio, ruido de fondo.

Una posibilidad relativamente simple es elegir características espectrales y temporales como son: energía, tasa de cruces por cero, centroide espectral, ancho de banda, energía relativa en subbandas. Estas características se normalizan y forman un vector de dimensión N. Elegir un tiempo de segmento, por ejemplo 2 o 3 segundos, y una frecuencia de muestreo razonable. Conseguir una cantidad representativa de segmentos de cada una de las categorías. Ajustar algún modelo de clasificación tradicional con los segmentos de entrenamiento (k-NN o alguna red neuronal sencilla). Dada una muestra larga a analizar, separarla en segmentos y aplicar el modelo para deducir a qué categoría pertenece.

## Qué hace el sistema

1. **Decodificación y remuestreo** a 16 kHz mono con un FIR anti-alias de Kaiser (80 dB) diseñado en el proyecto (`src/filters.py`).
2. **Acondicionamiento:** pasa-altos Butterworth de orden 4 en 70 Hz y notch de 50 Hz con Q = 10, en secciones de segundo orden; fase nula offline y versión causal para el micrófono.
3. **Análisis de corto plazo:** frames de 30 ms cada 10 ms con Hann periódica (COLA) y FFT de 512 (hay una FFT radix-2 propia en `src/fft.py`).
4. **64 características** por ventana de 3 s: energía, ZCR, centroide, ancho de banda, subbandas, rolloff, planitud, flujo, 13 MFCC con banco mel hecho a mano, deltas y fracción de baja energía.
5. **Clasificador** elegido entre k-NN, MLP, SVM, Random Forest y Gradient Boosting por validación cruzada agrupada por archivo, con datos aumentados (mezclas a SNR, reverb, canal de celular con códecs reales).
6. **Segmentación:** ventana de 3 s cada 0,5 s, probabilidad por celda, suavizado con Viterbi, duración mínima e intervalos con confianza.
7. **Interfaz web local** con línea de tiempo, panel de explicación por segmento, modo en vivo con micrófono, pestaña de diseño de filtros, demo de aliasing y métricas.

La teoría y el porqué de cada decisión están en [`explicacion.md`](explicacion.md); los pasos para correr todo en [`GuiaDeUso.md`](GuiaDeUso.md).

## Uso rápido

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.server              # http://127.0.0.1:8000
.venv/bin/python -m src.segmenter grabacion.m4a
.venv/bin/python -m pytest -q
```

En Windows (PowerShell), con Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m src.server          # http://127.0.0.1:8000
.venv\Scripts\python -m src.segmenter grabacion.m4a
.venv\Scripts\python -m pytest -q
```

## Estructura

```text
src/            pipeline: config, filtros, FFT, características, dataset, augmentation,
                entrenamiento, segmentación, modo en vivo y servidor
scripts/        figuras de diseño, demo de fuga espectral, robustez, evaluación con
                grabaciones reales y la versión anterior reproducida para comparar
tests/          pytest
web/            interfaz (HTML, CSS y JS sin dependencias externas)
models/         modelo entrenado, métricas y versión anterior (legacy/)
features/       características de MUSAN ya extraídas
docs/           figuras, resultados y notas para el informe
data/real/      instrucciones para grabar las pruebas propias
```
