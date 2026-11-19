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
| Suavizado | Viterbi | λ = 2 nats por cambio (antes 6; ver 3.2) | MAP de un HMM con segmentos que duran | `segmenter.viterbi` |
| Duración mínima | absorción | 1,5 s | evita segmentos espurios | `segmenter.enforce_min_duration` |
| En vivo | HMM hacia adelante | paso 1 s, T² | versión causal del Viterbi | `live.LiveSession` |
| Silencio | definición por nivel (medido tras el filtro) | −75 a −36 dBFS, 9 tipos (incluye `agc_celular`) | el silencio real es ruido de fondo; con AGC de celular medido en −40 a −42 dBFS | `augment.make_silence` |
| Clases activas | piso | ruido < −34 dBFS y voz/música < −50 dBFS descartadas | franja de 2 dB entre ruido y silencio; voz y música se separan por espectro | `config.NOISE_MIN_DBFS`, `ACTIVE_MIN_DBFS` |
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

Fuentes: `models/metricas.json` (CV y prueba), `docs/resultados/robustez.json` (robustez y segmentación), `docs/resultados/eval_real.json` (grabaciones propias). Todo con la definición final de silencio (−75 a −36 dBFS medido tras el filtro, 9 tipos).

**Dataset:** 60 000 ventanas de 3 s, 15 000 por clase (5 pliegues × 4 clases × 3000). Por pliegue y clase activa: 1800 limpias + 1200 sintéticas; silencio 3000 sintéticas. Extracción: 16 min con 7 procesos.

**Validación cruzada (pliegues 0–3, 48 000 ventanas, 64 características):**

| Modelo | F1 macro | Accuracy | Recall música | Recall voz | Recall ruido | Recall silencio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| k-NN (k=15) | 0,8810 ± 0,0054 | 0,8831 | 0,864 | 0,970 | 0,741 | 0,958 |
| Random Forest | 0,9189 ± 0,0117 | 0,9190 | 0,898 | 0,955 | 0,852 | 0,972 |
| MLP (128-64) | 0,9338 ± 0,0072 | 0,9342 | 0,920 | 0,972 | 0,865 | 0,980 |
| SVM RBF | 0,9376 ± 0,0124 | 0,9379 | 0,929 | 0,972 | 0,876 | 0,975 |
| **Gradient Boosting** | **0,9440 ± 0,0108** | **0,9442** | 0,925 | 0,974 | 0,890 | 0,987 |

**Prueba final (pliegue 4, 12 000 ventanas, Gradient Boosting):** accuracy 0,9428, F1 macro 0,9424. Recall: música 0,938, voz 0,974, ruido 0,871, silencio 0,988. F1: música 0,925, voz 0,971, ruido 0,899, silencio 0,975. Modelo: 2,0 MB.

Matriz de confusión (filas = real, columnas = predicho; música, voz, ruido, silencio):

```text
musica    2814    49   124    13
voz         29  2923    43     5
ruido      237    51  2613    99
silencio     5     0    31  2964
```

Accuracy por tipo de muestra en la prueba (peores): ruido con fondo de voz/música 0,767; celular banda ancha sin códec 0,881; banda angosta sin códec 0,896; mezcla con ruido 0,900; AMR-NB 0,907; AAC 32k 0,916; reverb y ganancia 0,917; MP3 32k 0,922.

**Ablation (Gradient Boosting, CV):**

| Grupos | Dimensión | F1 macro | Recall ruido |
| :--- | :---: | :---: | :---: |
| base | 18 | 0,8805 ± 0,0127 | 0,778 |
| + espectrales | 24 | 0,9242 ± 0,0118 | 0,865 |
| + MFCC | 50 | 0,9375 ± 0,0110 | 0,882 |
| + deltas | 63 | 0,9399 ± 0,0112 | 0,884 |
| + baja energía | 64 | 0,9440 ± 0,0108 | 0,890 |
| solo MFCC + deltas | 39 | 0,9118 ± 0,0090 | 0,820 |

**Robustez (pliegue de prueba, 300 recortes por clase, mismos recortes para los dos sistemas; nuevo / anterior):**

| Condición | Nuevo | Anterior |
| :--- | :---: | :---: |
| voz y música limpias | 0,965 | 0,950 |
| SNR 20 dB | 0,955 | 0,893 |
| SNR 10 dB | 0,928 | 0,732 |
| SNR 5 dB | 0,823 | 0,605 |
| SNR 0 dB | 0,667 | 0,538 |
| SNR −5 dB | 0,462 | 0,452 |
| fs 16 kHz (4 clases) | 0,961 | 0,873 |
| 44,1 → 16 kHz | 0,960 | 0,871 |
| 11 025 Hz | 0,934 | 0,766 |
| 8000 Hz | 0,905 | 0,654 |
| 6000 Hz | 0,871 | 0,644 |
| 8000 Hz sin anti-alias | 0,891 | 0,673 |
| MP3 128k | 0,962 | 0,868 |
| MP3 32k | 0,959 | 0,867 |
| AAC 32k | 0,954 | 0,874 |
| Opus 16k | 0,948 | 0,873 |
| AMR-NB 12,2k | 0,871 | 0,663 |
| canal de celular completo | 0,939 | 0,742 |

Con AMR-NB, recall de ruido: nuevo 0,847, anterior 0,237. A 0 dB, recall de voz: nuevo 0,533, anterior 0,250.

Anti-alias: a 8 kHz, 0,905 con filtro contra 0,891 sin filtro (+1,4 puntos, del orden del error de medición ≈ 0,009 con 1200 recortes). Se justifica sobre todo por fidelidad de la señal (demo: alias de −3 dB contra −100 dB).

**Segmentación (30 secuencias continuas del pliegue de prueba, 6–10 tramos de 3–15 s, 244 tramos):**

| Variante | Acierto por celda | Segmentos | Borde mediano | Borde medio |
| :--- | :---: | :---: | :---: | :---: |
| sin suavizado | 0,870 | 282 | 0,86 s | 0,98 s |
| mediana | 0,874 | 265 | 0,89 s | 0,99 s |
| Viterbi λ=2 (por defecto) | 0,876 | 240 | 0,91 s | 1,20 s |
| Viterbi λ=4 | 0,873 | 218 | 0,95 s | 1,64 s |
| Viterbi λ=6 | 0,872 | 212 | 0,96 s | 1,73 s |
| Viterbi λ=10 | 0,867 | 206 | 0,98 s | 1,91 s |

λ = 2 se eligió mirando esta tabla (pliegue de prueba): su fila es algo optimista. Con la definición de silencio anterior el óptimo había sido λ = 6 (0,899, 225/238 segmentos).

**Latencia del modo en vivo** (servidor, modelo final, mediana de 60 bloques, 48 y 44,1 kHz): procesamiento por bloque de 250 ms ≈ 2,3 ms sin decisión y ≈ 14 ms con decisión; FIR anti-alias 3,2 ms; espera del remuestreador 3,2 ms. Reacción de la etiqueta a un cambio de clase: estimada en 2–3,5 s (paso de 1 s + ventana de 3 s + confirmación del HMM).

**Advertencias para no sobrevender:**
- El silencio sintético sale muy bien (0,988), pero el silencio real grabado con celular falla (ver 3.3).
- La comparación de prueba con la versión anterior no es directa (otra partición, conjunto más difícil); la comparación justa son las curvas de robustez y las grabaciones reales, que usan el mismo audio para los dos sistemas.
- GB y SVM están a 0,006 de F1, del orden de un desvío.

### 3.3. Grabaciones reales

Cuatro notas de voz de celular (Opus, 48 kHz, 20–25 s), por celdas de 0,5 s:

| Archivo | Predicción | Acierto |
| :--- | :--- | :---: |
| voz_01 | voz todo | 1,000 |
| ruido_01 | ruido todo | 1,000 |
| musica_01 | voz 0–8 s, música 8–20 s | 0,610 |
| silencio_01 | ruido todo | 0,000 |

Total 0,678 (sin suavizado 0,655; versión anterior 0,379). F1 macro 0,576.

Sin `silencio_01` (etiqueta ambigua: la grabación tenía algo de ruido de fondo según quien grabó; pico de −26 dBFS al final): 134 celdas, acierto 0,881 (sin suavizado 0,851; versión anterior 0,448), F1 macro 0,873 sobre voz, música y ruido. La etiqueta no se cambió; se informan las dos cifras.

Silencio: con el AGC del celular queda en −40 a −42 dBFS y con variación espectral en el tiempo (desvío de mfcc_1 de 2,9 contra 0,95 en el silencio sintético); es indistinguible de ruido ambiente bajo. Se probó subir el techo de nivel (−52 → −38), fijar el nivel tras el filtro, un silencio tipo AGC y otro armado con ambientes reales de MUSAN: la confianza en ruido bajó de 0,999 a 0,57 en el último experimento, sin cambiar la decisión. Las grabaciones se usaron para diagnosticar, así que ya no son una prueba independiente.
