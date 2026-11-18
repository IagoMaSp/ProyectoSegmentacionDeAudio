# Notas para el informe

Material de base para armar el informe de 2–3 carillas. No es el informe: solo decisiones, parámetros, figuras y métricas, con la fuente de cada número.

## 1. Decisiones y parámetros

| Etapa | Decisión | Valor | Justificación corta | Dónde |
| :--- | :--- | :--- | :--- | :--- |
| Muestreo | fs de trabajo | 16 kHz | Nyquist 8 kHz cubre formantes y fricativas; nativa de MUSAN | `config.SAMPLE_RATE` |
| Remuestreo | anti-alias | FIR Kaiser, 80 dB, transición 7,2–8 kHz | rechazo desde la nueva Nyquist; 303 coef. (48→16 kHz), 44 265 (44,1→16 kHz) | `filters.design_antialias` |
| Pasa-altos | Butterworth | orden 4, fc = 70 Hz | debajo de F0 mínima (~85 Hz), arriba de rumble/viento; −43,5 dB en 20 Hz, −0,24 dB en 100 Hz | `filters.design_highpass` |
| Notch | IIR 2.º orden | 50 Hz, Q = 10 (BW 5 Hz) | red de Uruguay; −14 dB en ±0,5 Hz por pasada; sin armónicos para no tocar F0 | `filters.design_notch` |
| Implementación | SOS | 3 biquads, radio máx. de polos 0,99902 | estabilidad numérica con polos cerca de z = 1 | `filters.design_conditioning` |
| Aplicación offline | sosfiltfilt | fase nula, \|H\|² | sin corrimiento temporal | `filters.condition_offline` |
| Aplicación en vivo | sosfilt ×2 con estado | \|H\|², fase no nula | causal y mismo módulo que offline | `filters.CausalConditioner` |
| Frame | Hann periódica | 30 ms / 10 ms, FFT 512 | cuasi-estacionariedad; COLA exacta (N/H = 3); rejilla entera con la ventana | `config.N_WIN`, `HOP_LENGTH`, `N_FFT` |
| FFT | radix-2 propia | DIT iterativa | verificada contra np.fft (< 1e−9) | `src/fft.py` |
| Características | 64 por ventana | base 18 + espectrales 6 + MFCC 26 + deltas 13 + baja energía 1 | ver ablation | `features.py` |
| Mel | banco a mano | 40 filtros, 20–8000 Hz, 13 MFCC, DCT-II ortonormal | escala perceptual, decorrelación | `features.mel_filterbank` |
| Ventana de decisión | deslizante | 3 s cada 0,5 s, cola incluida | contexto estadístico + resolución de 0,5 s | `config.WINDOW_HOP` |
| Suavizado | Viterbi | λ = 6 nats por cambio (ε ≈ 0,0074 por celda) | MAP de un HMM con segmentos que duran | `segmenter.viterbi` |
| Duración mínima | absorción | 1,5 s | evita segmentos espurios | `segmenter.enforce_min_duration` |
| En vivo | HMM hacia adelante | paso 1 s, T² | versión causal del Viterbi | `live.LiveSession` |
| Silencio | definición por nivel | −85 a −52 dBFS, 8 tipos | el silencio real es ruido de fondo | `augment.make_silence` |
| Clases activas | piso | ventanas < −50 dBFS descartadas | franja de 2 dB sin ejemplos entre clases | `config.ACTIVE_MIN_DBFS` |
| Augmentation | por pliegue | mezclas 0–20 dB, ruido con fondo 8–20 dB, reverb T60 0,15–0,7 s, canal de celular | robustez a lo que no se filtra | `dataset.synth_fold` |
| Canal celular | pasa-banda + códec | 300–3400 Hz con AMR-NB o 100–7000 Hz con MP3/AAC/Opus | caminos reales de un celular | `augment.phone_channel` |
| Validación | pliegues por archivo/fuente | 5 pliegues: 0–3 CV, 4 prueba | sin leakage, con varianza | `dataset.assign_folds` |
| Selección | criterio | F1 macro medio en CV | pesa igual las cuatro clases | `train.py` |

## 2. Figuras disponibles (`docs/figuras/`)

| Archivo | Contenido | Generado por |
| :--- | :--- | :--- |
| `acond_bode.png` | módulo y fase de pasa-altos, notch, cascada y filtfilt | `scripts/filter_design.py` |
| `acond_notch_detalle.png` | muesca en escala lineal 40–60 Hz | idem |
| `acond_polos_ceros.png` | plano z completo y zooms | idem |
| `acond_retardo_grupo.png` | retardo de grupo offline, causal y en vivo | idem |
| `acond_antialias_fir.png` | respuesta del FIR anti-alias para 48, 44,1 y 22,05 kHz | idem |
| `aliasing_tonos.png` | tonos 1/5/7 kHz llevados a 8 kHz con y sin anti-alias | idem |
| `aliasing_chirp.png` | barrido 0–24 kHz llevado a 16 kHz con y sin anti-alias | idem |
| `fuga_espectral.png` | rectangular contra Hann con un tono débil a −50 dB | `scripts/leakage_demo.py` |
| `comparacion_modelos.png` | F1 macro en CV de los cinco modelos | `src/train.py` |
| `ablation.png` | F1 macro por grupo de características | idem |
| `confusion_cv_*.png`, `confusion_prueba.png` | matrices de confusión | idem |
| `robustez_snr.png`, `robustez_fs.png`, `robustez_codec.png` | nuevo contra anterior | `scripts/eval_robustness.py` |
| `real_lineas.png` | líneas de tiempo real contra predicho en grabaciones propias | `scripts/eval_real.py` |

Valores de la demo de aliasing y del diseño en `docs/figuras/acond_resumen.json`: alias de 5 kHz en 3 kHz a −3,1 dB sin filtro y −100,2 dB con filtro. Fuga espectral: el tono débil de 2,5 kHz (−50 dB) queda tapado por la fuga de la rectangular (−45,5 dB en esa frecuencia) y se ve en −50,0 dB con Hann.

## 3. Métricas

### 3.1. Versión anterior (línea base)

Medido al inicio de este trabajo sobre `features/musan_features.npz` del commit `35b14de` (partición 80/20 por archivo, 35 036 ventanas de prueba), con los modelos que estaban en `models/` (`models/legacy/metricas_legacy.json`):

| Modelo | Accuracy | F1 macro | Recall música | Recall voz | Recall ruido | Recall silencio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MLP (en producción) | 0,9642 | 0,9069 | 0,9476 | 0,9878 | 0,6589 | 0,9957 |
| k-NN | 0,9286 | 0,8083 | 0,9234 | 0,9849 | 0,2835 | 0,9510 |

Observación: el ruido era la clase débil (5 477 ventanas de entrenamiento contra 57 176 de voz) y se confundía con música y silencio.

### 3.2. Sistema nuevo

Fuentes: `models/metricas.json` (CV y prueba), `docs/resultados/robustez.json` (robustez y segmentación).

**Dataset:** 60 000 ventanas de 3 s, 15 000 por clase (5 pliegues × 4 clases × 3000). Por pliegue y clase activa: 1800 limpias + 1200 sintéticas; silencio 3000 sintéticas. Extracción: 18 min con 7 procesos.

**Validación cruzada (pliegues 0–3, 48 000 ventanas, 64 características):**

| Modelo | F1 macro | Accuracy | Recall música | Recall voz | Recall ruido | Recall silencio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| k-NN (k=15) | 0,8986 ± 0,0074 | 0,8995 | 0,869 | 0,968 | 0,800 | 0,960 |
| Random Forest | 0,9299 ± 0,0102 | 0,9300 | 0,905 | 0,954 | 0,866 | 0,996 |
| MLP (128-64) | 0,9429 ± 0,0075 | 0,9432 | 0,932 | 0,974 | 0,876 | 0,990 |
| SVM RBF | 0,9457 ± 0,0077 | 0,9459 | 0,934 | 0,971 | 0,890 | 0,988 |
| **Gradient Boosting** | **0,9486 ± 0,0063** | **0,9488** | 0,926 | 0,972 | 0,899 | 0,998 |

**Prueba final (pliegue 4, 12 000 ventanas, Gradient Boosting):** accuracy 0,9550, F1 macro 0,9548. Recall: música 0,948, voz 0,976, ruido 0,897, silencio 0,999. F1: música 0,936, voz 0,972, ruido 0,919, silencio 0,992.

Matriz de confusión (filas = real, columnas = predicho; música, voz, ruido, silencio):

```text
musica    2845    33   117     5
voz         27  2927    45     1
ruido      207    62  2691    40
silencio     0     0     3  2997
```

Accuracy por tipo de muestra en la prueba (peores): ruido con fondo de voz/música 0,743; celular banda angosta sin códec 0,909; Opus 16k 0,910; MP3 32k 0,911; mezcla con ruido 0,912; AMR-NB 0,917; limpias 0,964; silencios sintéticos ≥ 0,994.

**Ablation (Gradient Boosting, CV):**

| Grupos | Dimensión | F1 macro | Recall ruido |
| :--- | :---: | :---: | :---: |
| base | 18 | 0,8980 ± 0,0124 | 0,815 |
| + espectrales | 24 | 0,9328 ± 0,0089 | 0,882 |
| + MFCC | 50 | 0,9435 ± 0,0082 | 0,893 |
| + deltas | 63 | 0,9478 ± 0,0077 | 0,899 |
| + baja energía | 64 | 0,9486 ± 0,0063 | 0,899 |
| solo MFCC + deltas | 39 | 0,9228 ± 0,0067 | 0,842 |

**Robustez (pliegue de prueba, 300 recortes por clase; nuevo / anterior):**

| Condición | Nuevo | Anterior |
| :--- | :---: | :---: |
| voz y música limpias | 0,957 | 0,942 |
| SNR 20 dB | 0,953 | 0,888 |
| SNR 10 dB | 0,910 | 0,717 |
| SNR 5 dB | 0,842 | 0,608 |
| SNR 0 dB | 0,640 | 0,532 |
| SNR −5 dB | 0,450 | 0,437 |
| fs 16 kHz (4 clases) | 0,964 | 0,909 |
| 44,1 → 16 kHz | 0,964 | 0,894 |
| 11 025 Hz | 0,946 | 0,784 |
| 8000 Hz | 0,918 | 0,670 |
| 6000 Hz | 0,892 | 0,676 |
| 8000 Hz sin anti-alias | 0,918 | 0,695 |
| MP3 128k | 0,964 | 0,889 |
| MP3 32k | 0,961 | 0,893 |
| AAC 32k | 0,958 | 0,912 |
| Opus 16k | 0,963 | 0,911 |
| AMR-NB 12,2k | 0,907 | 0,682 |
| canal de celular completo | 0,943 | 0,736 |

Con AMR-NB, recall de ruido: nuevo 0,847, anterior 0,213. A 0 dB, recall de voz: nuevo 0,527, anterior 0,293.

Observación para el informe: con y sin anti-alias a 8 kHz la accuracy es la misma (0,918). El anti-alias no mejora la clasificación medida en MUSAN; se justifica por fidelidad de la señal (demo: alias de −3 dB contra −100 dB).

**Segmentación (30 secuencias continuas del pliegue de prueba, 6–10 tramos de 3–15 s, 238 tramos):**

| Variante | Acierto por celda | Segmentos | Borde mediano | Borde medio |
| :--- | :---: | :---: | :---: | :---: |
| sin suavizado | 0,869 | 314 | 0,67 s | 0,79 s |
| mediana | 0,870 | 299 | 0,76 s | 0,83 s |
| Viterbi λ=2 | 0,885 | 258 | 0,77 s | 1,03 s |
| Viterbi λ=4 | 0,894 | 235 | 0,83 s | 1,18 s |
| Viterbi λ=6 (por defecto) | 0,899 | 225 | 0,83 s | 1,27 s |
| Viterbi λ=10 | 0,897 | 216 | 0,87 s | 1,60 s |

**Latencia del modo en vivo** (medida en el servidor con el modelo definitivo, mediana de 60 bloques, 48 y 44,1 kHz): procesamiento por bloque de 250 ms ≈ 2,5 ms sin decisión y ≈ 14,7 ms con decisión; FIR anti-alias 3,2 ms; espera del remuestreador 3,2 ms. Reacción de la etiqueta a un cambio de clase: estimada en 2–3,5 s (paso de 1 s + ventana de 3 s + confirmación del HMM).

**Advertencias para no sobrevender:**
- El silencio sintético sale casi perfecto (≥ 0,994) porque está definido por nivel; la prueba real para esa clase son las grabaciones de `data/real/`.
- La comparación de prueba con la versión anterior no es directa (otra partición y otro conjunto, más difícil); la comparación justa son las curvas de robustez, que usan los mismos recortes para los dos sistemas.
- GB y SVM son equivalentes dentro de un desvío.

### 3.3. Grabaciones reales

Pendiente: grabar según `data/real/LEEME.md` y correr `python -m scripts.eval_real`. Los resultados quedan en `docs/resultados/eval_real.json`.
