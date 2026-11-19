# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Estudiantes, docentes y evaluadores de la materia **Muestreo y Procesamiento de Señales**, que usan la herramienta para demostrar, inspeccionar y defender cómo el sistema segmenta distintos tipos de señal en el tiempo, y para mostrar en vivo las decisiones de diseño (filtros, muestreo, características).

## Product Purpose

Subir cualquier archivo de audio o video, o hablarle al micrófono, y ver su segmentación en **música**, **voz**, **ruido** y **silencio** con resolución de 0,5 s; escuchar cada segmento y entender por qué se clasificó así. Además, mostrar con datos del propio sistema el diseño del acondicionamiento, el efecto del aliasing y las métricas de evaluación.

## Positioning

Interfaz web liviana servida por `src/server.py` (biblioteca estándar de Python), sin dependencias de internet. ffmpeg solo decodifica; el remuestreo, el filtrado, las características y la segmentación son del proyecto.

## Operating Context

- Ejecución local con `python -m src.server` (por defecto `http://127.0.0.1:8000`).
- Uso en la defensa (8 minutos, proyector), en pruebas de laboratorio y con grabaciones de celular.
- Rutas: `GET /` y estáticos; `POST /api/segmentar`; `GET /api/explicar`, `/api/audio`, `/api/filtros`, `/api/aliasing`, `/api/aliasing.wav`, `/api/metricas`, `/api/estado`; `POST /api/vivo` y `/api/vivo/fin`.

## Capabilities and Constraints

- Formatos: cualquiera que lea ffmpeg (wav, mp3, m4a, ogg, flac, opus, amr, video con audio), cualquier frecuencia de muestreo. Máximo 200 MB.
- Ventana de clasificación de 3 s con salto configurable (0,25 / 0,5 / 1 / 3 s); la cola final se clasifica con una ventana alineada al final.
- Suavizado configurable: Viterbi (penalidad en nats), mediana o ninguno; duración mínima de segmento.
- Panel por segmento: forma de onda, espectrograma acondicionado, curvas de RMS, ZCR, centroide y planitud, probabilidades, PCA 2D con vecinos y rasgos determinantes.
- Modo en vivo: micrófono sin procesamiento del navegador, espectrograma en tiempo real y etiqueta cada 1 s con filtrado HMM.
- Exportación JSON, CSV y SRT.
- Categorías fijas: `musica`, `voz`, `ruido`, `silencio`.

## Product Principles

1. **Claridad inmediata de la señal:** la relación entre audio, categorías e intervalos se lee de un vistazo.
2. **Precisión académica sin ruido visual:** cada número que se muestra sale del sistema (parámetros, métricas medidas), con unidades.
3. **Feedback honesto:** estados de carga, errores de decodificación, confianza por segmento y aviso si no hay modelo.
4. **Explicar, no solo mostrar:** cada decisión del sistema (filtro, ventana, clase) se puede inspeccionar desde la interfaz.
5. **Un clic para verificar:** escuchar cualquier segmento o comparar antes/después del filtro toma un clic.
