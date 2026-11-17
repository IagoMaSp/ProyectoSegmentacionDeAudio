# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Estudiantes, docentes y evaluadores en el contexto de la materia **Muestreo** (proyecto académico de segmentación automática de audio), que suben grabaciones para demostrar, inspeccionar y validar cómo el modelo clasifica distintos tipos de señal en el tiempo.

## Product Purpose

Permitir subir cualquier archivo de audio (o video con pista de audio) y visualizar de forma inmediata y clara su segmentación automática en bloques de 3 segundos clasificados en cuatro categorías: **música**, **voz**, **ruido** y **silencio**, permitiendo reproducir y saltar a cada segmento detectado.

## Positioning

Interfaz web liviana conectada directamente al servidor local de inferencia (`src/server.py`), que decodifica múltiples formatos vía `ffmpeg` a 16 kHz mono, extrae características espectrales/temporales y muestra la línea de tiempo y tabla de intervalos devueltos por el modelo entrenado.

## Operating Context

- Ejecución local mediante `python -m src.server` (por defecto en `http://127.0.0.1:8000`).
- Uso en presentaciones en vivo, pruebas de laboratorio y evaluación de muestras de audio variadas (podcast, música, grabaciones con ruido ambiente, silencios).
- El servidor actual expone `GET /` (sirviendo `web/index.html`) y `POST /segment` (recibe el archivo en crudo y devuelve `{ duration, segments: [{ start_time, end_time, label }] }`), con límite de 200 MB.

## Capabilities and Constraints

- Formatos soportados: `audio/*, video/*` decodificados por `ffmpeg` en el backend (wav, mp3, ogg, flac, m4a, opus, etc.).
- Granularidad fija de clasificación: bloques de 3 segundos; los segundos finales que no completan un bloque de 3 s quedan sin clasificar y se informa al usuario.
- Categorías fijas del modelo: `musica`, `voz`, `ruido`, `silencio`.
- Tamaño máximo de archivo: 200 MB.
- Interacción clave: carga de archivo, procesamiento con feedback de estado/error, reproducción de audio integrada y salto instantáneo al hacer clic en un segmento.
- Se permite sumar librerías si aportan valor, manteniendo compatibilidad directa con el servidor Python existente.

## Product Principles

1. **Claridad inmediata de la señal**: la relación entre el audio en el tiempo, las categorías detectadas y los intervalos exactos debe leerse de un vistazo.
2. **Precisión académica sin ruido visual**: presentar métricas, duraciones y etiquetas con rigor técnico, sin adornos que distraigan de los resultados del modelo.
3. **Feedback honesto en cada estado**: comunicar claramente la carga, el procesamiento en curso, bloques no clasificados al final y errores de decodificación o tamaño.
4. **Exploración directa audio-segmento**: escuchar y verificar cualquier segmento clasificado debe tomar un solo clic.
