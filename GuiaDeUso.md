---
title: "Guia de Uso del Sistema de Segmentacion y Clasificacion de Audio"
date: "2026-11-06"
---

# Guia de Uso

## 1. Instalar las dependencias en el entorno virtual

Desde la raiz del proyecto (`/home/iago/Facultad/Muestreo`), instala los paquetes requeridos dentro del entorno virtual `.venv`:

```bash
.venv/bin/pip install numpy scipy scikit-learn joblib
```

---

## 2. Ubicar el dataset MUSAN

Por defecto, `src/config.py` busca el dataset en el directorio `data/musan/`, aunque tambien se puede especificar cualquier otra ruta por linea de comandos.

La estructura esperada dentro de la carpeta del dataset es la original de MUSAN (los archivos `.wav` pueden estar anidados en subdirectorios internos, ya que `collect_wav_files` realiza una busqueda recursiva):

```text
data/musan/
├── music/
├── speech/
└── noise/
```

El sistema tambien reconoce los nombres de carpetas en espanol (`musica/`, `voz/`, `ruido/`).

---

## 3. Caracteristicas precalculadas (`features/musan_features.npz`)

El repositorio incluye `features/musan_features.npz` (~24 MB) con los vectores de 18 caracteristicas ya extraidos de MUSAN (particiones train/test, etiquetas, nombres de caracteristicas y parametros usados). Gracias a esto **no hace falta descargar el dataset de audio** para entrenar: `src/train.py` usa el `.npz` automaticamente si existe.

Para regenerarlo desde los `.wav` (requiere `data/musan/`, tarda unos 4 minutos):

```bash
.venv/bin/python -m src.build_features
```

Contenido del archivo:

```python
import numpy as np
d = np.load("features/musan_features.npz")
d["X_train"], d["X_test"], d["y_train"], d["y_test"], d["feature_names"], d["categories"]
```

---

## 4. Entrenar y evaluar los modelos (`src/train.py`)

Para entrenar (usa `features/musan_features.npz` si existe; si no, extrae desde `data/musan`):

```bash
.venv/bin/python -m src.train
```

Para indicar una ruta personalizada al dataset o limitar la cantidad maxima de segmentos de 3 segundos por categoria (por ejemplo, `500` por clase para una ejecucion rapida):

```bash
.venv/bin/python -m src.train --data-dir /ruta/a/musan --max-segments 500
```

Para forzar la extraccion desde el audio ignorando el `.npz`, agrega `--from-audio`. (`--max-segments` solo aplica al extraer desde audio.)

Al extraer desde audio, el pipeline realiza automaticamente las siguientes etapas:
1. Separa los archivos `.wav` en conjuntos de entrenamiento ($80\%$) y prueba ($20\%$) a nivel de archivo para evitar data leakage.
2. Genera la cuarta categoria (`silencio`) de forma independiente para cada particion combinando tramas de voz de baja energia y segmentos de ruido escalados entre `0.001` y `0.01`.
3. Extrae los vectores de 18 caracteristicas temporales y espectrales por cada segmento de 3 segundos.
4. Ajusta el `StandardScaler` unicamente sobre la particion de entrenamiento y entrena los modelos `KNeighborsClassifier` y `MLPClassifier`.
5. Muestra en consola el `Accuracy`, `F1 Macro` y el reporte de clasificacion detallado por categoria.
6. Persiste los artefactos resultantes dentro del directorio `models/`:
   - `models/scaler.joblib`
   - `models/knn_model.joblib`
   - `models/mlp_model.joblib`
   - `models/best_model.joblib`

---

## 5. Probar la segmentacion sobre un archivo `.wav` (`src/segmenter.py`)

Una vez entrenados y persistidos los modelos, se puede segmentar cualquier archivo `.wav` mono a $16000\text{ Hz}$ de duracion arbitraria desde la terminal:

```bash
.venv/bin/python -m src.segmenter /ruta/al/archivo_prueba.wav
```

La salida mostrara la prediccion para cada bloque consecutivo de 3 segundos:

```text
Inicio (s) | Fin (s)    | Categoria
--------------------------------------
      0.00 |       3.00 | silencio
      3.00 |       6.00 | voz
      6.00 |       9.00 | voz
      9.00 |      12.00 | musica
```

Para fusionar bloques consecutivos que tengan la misma categoria predicha en intervalos continuos, agrega el argumento `--merge`:

```bash
.venv/bin/python -m src.segmenter /ruta/al/archivo_prueba.wav --merge
```

Para utilizar un modelo especifico (`knn_model.joblib` o `mlp_model.joblib`) en lugar de `best_model.joblib`:

```bash
.venv/bin/python -m src.segmenter /ruta/al/archivo_prueba.wav --model-path models/knn_model.joblib
```
