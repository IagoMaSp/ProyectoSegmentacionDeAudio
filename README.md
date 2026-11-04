# Segmentación de audio

## Objetivo

Dada una señal de audio, segmentarla automáticamente en categorías, como pueden ser música, voz hablada, silencio, ruido de fondo.

## Guía

Una posibilidad relativamente simple es elegir características espectrales y temporales como son: energía, tasa de cruces por cero, centroide espectral, ancho de banda, energía relativa en subbandas.

Estas características se normalizan y forman un vector de dimensión N (N puede ser 4 o más según la cantidad de características).

Elegir un tiempo de segmento, por ejemplo 2 o 3 segundos, y una frecuencia de muestreo razonable. Conseguir una cantidad representativa de segmentos de cada una de las categorías (segmentos de entrenamiento).

Ajustar algún modelo de clasificación tradicional con los segmentos de entrenamiento. Puede ser k-NN, o alguna red neuronal sencilla.

Dada una muestra larga a analizar, separarla en segmentos y aplicar el modelo para deducir a qué categoría pertenece.
