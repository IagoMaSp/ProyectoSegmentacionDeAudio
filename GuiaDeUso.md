---
title: "Guia de Uso del Sistema de Segmentacion y Clasificacion de Audio"
---

# Guia de Uso

Todos los comandos se corren desde la raiz del proyecto.

## 1. Instalar las dependencias

Se necesita Python 3.12. Para bajar el proyecto: `git clone -b mejoras https://github.com/IagoMaSp/ProyectoSegmentacionDeAudio.git`, o desde GitHub con **Code → Download ZIP**.

Mac / Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

En Windows todos los comandos de esta guia son iguales cambiando `.venv/bin/python` por `.venv\Scripts\python`. Si `py` no existe, instalar Python desde python.org marcando la opcion "Add python.exe to PATH" y usar `python -m venv .venv`.

`imageio-ffmpeg` trae un binario de `ffmpeg`, asi que no hace falta instalarlo aparte; si hay uno en el sistema se usa ese. Se necesita para leer formatos comprimidos (mp3, m4a, ogg, amr) y para la simulacion de codecs.

---

## 2. Usar el modelo ya entrenado

El repositorio incluye `models/modelo.joblib` y sus metricas, asi que la interfaz y la segmentacion por linea de comandos funcionan sin descargar nada.

### 2.1. Interfaz web

```bash
.venv/bin/python -m src.server            # http://127.0.0.1:8000
.venv/bin/python -m src.server --port 9000
```

Windows (PowerShell):

```powershell
.venv\Scripts\python -m src.server        # http://127.0.0.1:8000
.venv\Scripts\python -m src.server --port 9000
```

Abrir la direccion en el navegador y cortar el servidor con `Ctrl+C`.

Funciona sin conexion a internet. Pestanas:

- **Archivo:** subir un audio o video (cualquier formato y frecuencia, hasta 200 MB). Se puede elegir salto de ventana, tipo de suavizado, penalidad de cambio y duracion minima. Clic en un segmento: lo reproduce y abre el panel con espectrograma, curvas, probabilidades, PCA con vecinos y rasgos que mas pesaron. Exporta JSON, CSV y SRT.
- **En vivo:** clasificacion con el microfono. El navegador pide permiso; en `127.0.0.1` funciona sin HTTPS.
- **Acondicionamiento:** Bode, polos y ceros y retardo de grupo del filtro (fijos, no dependen del audio). Abajo, con la ultima senal procesada: su espectro antes (A) y despues (B) del filtro, un resumen de cuanta energia se quito bajo 70 Hz y en 50 Hz, y la escucha A/B.
- **Aliasing:** tono o barrido llevado de 48 kHz a 6–16 kHz con y sin anti-alias, con audio.
- **Resultados:** validacion cruzada, prueba final, ablation, robustez y grabaciones reales.

### 2.2. Linea de comandos

```bash
.venv/bin/python -m src.segmenter grabacion.m4a
.venv/bin/python -m src.segmenter grabacion.wav --suavizado mediana --salto 1
.venv/bin/python -m src.segmenter grabacion.wav --penalidad 6 --duracion-minima 3
```

Salida real para una secuencia armada con audio del pliegue de prueba, convertida a m4a a 44.1 kHz (los tramos reales eran silencio 0–10.3 s, ruido 10.3–23.3 s, voz 23.3–29.4 s, ruido 29.4–37.1 s, silencio 37.1–41.8 s, ruido 41.8–46.2 s y musica 46.2–50.6 s):

```text
Duracion: 50.59 s | suavizado: viterbi
Inicio (s) | Fin (s)    | Categoria | Confianza
--------------------------------------------------
      0.00 |       9.00 | silencio  | 0.97
      9.00 |      22.50 | ruido     | 0.84
     22.50 |      29.00 | voz       | 0.86
     29.00 |      36.00 | ruido     | 0.89
     36.00 |      43.00 | silencio  | 0.87
     43.00 |      46.00 | ruido     | 0.74
     46.00 |      50.59 | musica    | 0.93
```

Los bordes quedan dentro de 1.3 s de los reales: es la resolucion que permite una ventana de 3 s. Ojo: las grabaciones de celular con control automatico de ganancia (por ejemplo notas de voz de mensajeria) suben el silencio de la pieza a unos −41 dBFS y el sistema lo marca como ruido; ver `explicacion.md`, seccion 9.4.

---

## 3. Regenerar caracteristicas y reentrenar

### 3.1. Dataset MUSAN

Descargar MUSAN (11 GB) de https://www.openslr.org/17/ y descomprimirlo en `data/`:

```bash
mkdir -p data && cd data
curl -L -O https://www.openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz && cd ..
```

Estructura esperada (la busqueda de `.wav` es recursiva; tambien se aceptan carpetas `musica/`, `voz/`, `ruido/`):

```text
data/musan/
├── music/
├── speech/
└── noise/
```

### 3.2. Extraer caracteristicas (`src/build_features.py`)

```bash
.venv/bin/python -m src.build_features            # usa todos los nucleos menos uno
.venv/bin/python -m src.build_features --workers 4
```

Asigna cada archivo a uno de 5 pliegues, acondiciona y extrae ventanas de 3 s, genera silencio sintetico y augmentation por pliegue, y guarda:

- `features/musan_features.npz`: `X` (ventanas x 64), `y`, `fold`, `group`, `kind` (tipo de muestra) y los nombres de caracteristicas.
- `data/banco_prueba.npz`: recortes crudos del pliegue de prueba para las curvas de robustez (no se versiona).

### 3.3. Entrenar (`src/train.py`)

```bash
.venv/bin/python -m src.train
.venv/bin/python -m src.train --modelos svm,rf,gb --sin-ablation   # mas rapido
```

Compara los cinco clasificadores en validacion cruzada (pliegues 0–3), elige por F1 macro, hace la ablation de grupos de caracteristicas, evalua una vez en el pliegue 4 y guarda:

- `models/modelo.joblib`: modelo con su escalado, PCA y muestra de entrenamiento para el panel *por que*.
- `models/metricas.json`: todas las metricas.
- `docs/figuras/confusion_*.png`, `comparacion_modelos.png`, `ablation.png`.

---

## 4. Evaluacion

```bash
.venv/bin/python -m scripts.eval_robustness    # SNR, frecuencia de muestreo, codecs y segmentacion
.venv/bin/python -m scripts.eval_real          # grabaciones propias en data/real/
```

Para las grabaciones propias, ver `data/real/LEEME.md` (que grabar y como etiquetar).

## 5. Figuras de teoria

```bash
.venv/bin/python -m scripts.filter_design      # Bode, polos y ceros, retardo de grupo, aliasing
.venv/bin/python -m scripts.leakage_demo       # fuga espectral rectangular contra Hann
```

Quedan en `docs/figuras/`.

## 6. Tests

```bash
.venv/bin/python -m pytest -q
```

Cubren: notch con menos de −40 dB en 50 Hz, banda de paso plana, estabilidad, fase nula de filtfilt, equivalencia entre filtrado causal doble y offline, anti-alias, remuestreo por bloques, FFT propia contra `np.fft`, COLA, banco mel, DCT, conteo de frames, equivalencia entrenamiento/inferencia, Viterbi, mediana, duracion minima y cobertura de la cola.
