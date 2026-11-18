---
title: "Fundamentos Teoricos y Procedimiento del Sistema de Segmentacion y Clasificacion de Audio"
date: "2026-11-18"
---

# Sistema de Segmentacion y Clasificacion Automatica de Audio

Este documento explica la teoria detras de cada etapa del sistema y el porque de cada decision de diseno. Cada seccion indica el modulo de `src/` (o el script de `scripts/`) donde se implementa. Los valores numericos de los parametros estan centralizados en `src/config.py`; los resultados medidos estan en `models/metricas.json`, `docs/resultados/` y resumidos en `docs/NOTAS_INFORME.md`.

## 1. Descripcion del Problema y de las Senales

### 1.1. Objetivo

Dada una grabacion de duracion arbitraria (un archivo o el microfono en vivo), dividirla en intervalos y asignar a cada uno una de cuatro categorias: **musica**, **voz**, **ruido** o **silencio**. La salida es una lista de intervalos $[t_{\text{inicio}}, t_{\text{fin}})$ con su clase y una confianza.

### 1.2. Tipo de senales

Todas las senales son de audio: unidimensionales, reales, de banda base, y en el sistema siempre discretas. Llegan digitalizadas por el conversor A/D de un celular o una placa de sonido (tipicamente 44.1 o 48 kHz, 16 bits) o desde un archivo comprimido (mp3, aac, opus, amr). El sistema las lleva a una representacion comun: mono, $f_s = 16\text{ kHz}$, punto flotante en $[-1, 1)$.

### 1.3. Senal de interes en tiempo y en espectro

| Clase | En el tiempo | En el espectro |
| :--- | :--- | :--- |
| Voz | Modulacion silabica de amplitud a 3–6 Hz; alternancia de tramos sonoros (casi periodicos, $F_0 \approx 85$–$255\text{ Hz}$) y sordos (fricativas, parecidas a ruido); pausas cortas. | Armonicos de $F_0$ con envolvente de formantes ($F_1$ 300–900 Hz, $F_2$ 900–2500 Hz); energia util hasta 4 kHz y fricativas hasta 8 kHz. Cambia rapido de frame a frame. |
| Musica | Energia mas sostenida, ritmo (periodicidad de 1–3 Hz), pocas pausas. | Lineas armonicas estables durante decenas de ms, banda ancha (bajo + percusion + agudos); espectro con estructura pero que cambia mas lento que la voz. |
| Ruido | Estacionario (ventilador, lluvia) o impulsivo (golpes, transito). Nivel alto. | Sin estructura armonica, espectro "plano" o de pendiente suave; alta planitud espectral. |
| Silencio | Nivel bajo y sin eventos (por debajo de unos $-36\text{ dBFS}$); en la practica es ruido de fondo de la sala, del preamplificador o ceros digitales con dither. | El espectro del ruido de fondo: rosa o marron en una sala, blanco con enfasis en agudos para el siseo del microfono. |

La clase silencio no se distingue del ruido por la forma del espectro sino por el **nivel**: esa es una decision de diseno explicita (seccion 6.3).

### 1.4. Ruidos e interferencias que hay que tratar

1. **Continua (DC offset)** del conversor A/D y **rumble** por debajo de 50 Hz (manipulacion del celular, viento, aire acondicionado). Ensucian la energia, la tasa de cruces por cero y el bin $k = 0$ de la FFT.
2. **Zumbido de red de 50 Hz** (Uruguay) y sus armonicos, en grabaciones con equipos enchufados o cables sin blindaje.
3. **Aliasing** si se reduce la frecuencia de muestreo sin filtrar (por ejemplo, de 48 kHz a 16 kHz).
4. **Canal del celular:** respuesta pasa-banda del microfono, reverberacion de la habitacion, control automatico de ganancia (AGC) que puede saturar, y codecs con perdida (AMR-NB en llamadas, AAC en notas de voz, Opus en mensajeria).
5. **Ruido aditivo de fondo** sobre la voz o la musica (SNR variable).

Los puntos 1–3 se atacan con **acondicionamiento** (secciones 2 y 3). Los puntos 4–5 no se pueden filtrar sin destruir la senal de interes, asi que se atacan haciendo el clasificador **robusto**: entrenandolo con esas degradaciones simuladas (seccion 6) y midiendo su efecto (seccion 9).

### 1.5. Arquitectura general

```text
archivo / microfono
   -> decodificacion (ffmpeg, frecuencia original)              src/preprocessing.py
   -> remuestreo a 16 kHz con FIR anti-alias                    src/filters.py
   -> pasa-altos 70 Hz + notch 50 Hz (fase nula offline)        src/filters.py
   -> frames de 30 ms cada 10 ms, Hann, FFT de 512              src/features.py, src/fft.py
   -> 38 descriptores por frame
   -> ventanas de 3 s cada 0.5 s: 64 caracteristicas            src/features.py
   -> clasificador -> P(clase | ventana)                        src/train.py
   -> probabilidad por celda de 0.5 s, Viterbi, duracion minima src/segmenter.py
   -> intervalos con confianza                                  src/server.py, web/
```

---

## 2. Muestreo, Nyquist, Aliasing y Remuestreo (`src/filters.py`, `src/preprocessing.py`)

### 2.1. Muestreo ideal y teorema de Nyquist

Muestrear $x(t)$ con periodo $T_s = 1/f_s$ equivale a multiplicarla por un tren de impulsos. En frecuencia, el espectro queda replicado cada $f_s$:

$$X_s(f) = \frac{1}{T_s} \sum_{k=-\infty}^{\infty} X(f - k f_s)$$

Si $x(t)$ es de banda limitada a $B$ y $f_s > 2B$, las replicas no se superponen y $x(t)$ se recupera exactamente con un pasa-bajos ideal (teorema de Nyquist-Shannon). La mitad de la frecuencia de muestreo, $f_N = f_s/2$, es la **frecuencia de Nyquist**.

**Eleccion de $f_s = 16\text{ kHz}$** (`SAMPLE_RATE`): $f_N = 8\text{ kHz}$ cubre los formantes de la voz y la mayor parte de la energia que distingue musica de ruido. Bajar a 8 kHz (telefonia) pierde fricativas y brillo; subir a 44.1 kHz multiplica el costo por 2.75 sin mejorar la discriminacion de estas cuatro clases. Ademas es la tasa nativa de MUSAN, por lo que el entrenamiento no requiere remuestrear.

### 2.2. Aliasing

Si hay energia por encima de $f_N$, las replicas se solapan y una componente de frecuencia $f > f_N$ aparece en

$$f_{\text{alias}} = \left| f - k f_s \right|, \quad k = \operatorname{round}(f / f_s)$$

Es irreversible: una vez muestreada, el alias es indistinguible de una componente real. Ejemplo de la demo (`scripts/filter_design.py`, pestana *Aliasing*): un tono de 5 kHz llevado a $f_s = 8\text{ kHz}$ ($f_N = 4\text{ kHz}$) sin filtro aparece en $|5000 - 8000| = 3\text{ kHz}$ con practicamente el mismo nivel (−3.1 dB medido en la demo, `docs/figuras/acond_resumen.json`); con el anti-alias el residuo en 3 kHz queda en −100 dB. Un barrido de 0 a 24 kHz llevado a 16 kHz sin filtro "rebota" en 8 kHz (`docs/figuras/aliasing_chirp.png`).

### 2.3. Remuestreo racional con anti-alias explicito

Para pasar de $f_{\text{in}}$ a $f_{\text{out}}$ se escribe $f_{\text{out}}/f_{\text{in}} = L/M$ irreducible (48 kHz → 16 kHz: $L=1, M=3$; 44.1 kHz → 16 kHz: $L=160, M=441$) y se hace:

1. **Interpolacion por $L$:** se intercalan $L-1$ ceros entre muestras. Aparecen imagenes espectrales en multiplos de $f_{\text{in}}$.
2. **Filtro pasa-bajos $h[n]$** a la tasa intermedia $L f_{\text{in}}$: elimina las imagenes y, a la vez, todo lo que supere la nueva Nyquist. Es el filtro **anti-alias** (y anti-imagen).
3. **Diezmado por $M$:** se toma una muestra de cada $M$.

`scipy.signal.resample_poly` hace los tres pasos con una implementacion polifasica (no calcula las muestras que el diezmado descarta). El proyecto no usa el filtro por defecto sino uno disenado explicitamente (`design_antialias`):

- **Tipo:** FIR de fase lineal (simetrico, largo impar). No distorsiona la forma de onda en la banda de paso y su retardo es constante, que `resample_poly` compensa.
- **Ventana de Kaiser:** permite fijar la atenuacion de la banda de rechazo $A$ y obtener el orden minimo. Las formulas de Kaiser dan $\beta = 0.1102 (A - 8.7)$ para $A > 50\text{ dB}$ y un largo $N \approx \frac{A - 7.95}{2.285 \, \Delta\omega}$, con $\Delta\omega$ el ancho de transicion en rad/muestra.
- **Especificacion** (`AA_ATTENUATION_DB = 80`, `AA_TRANSITION_FRACTION = 0.10`): rechazo de 80 dB a partir de la nueva Nyquist (8 kHz) y banda de transicion de 7.2 a 8 kHz. 80 dB alcanza: un alias atenuado 80 dB queda por debajo del piso de cuantizacion practico de una grabacion de celular. La banda de rechazo empieza **exactamente** en $f_N$, asi nada por encima de 8 kHz puede plegarse; la transicion queda dentro de la banda util, donde lo unico que se pierde es la parte alta (7.2–8 kHz) del espectro, sin aliasing.
- **Resultado:** 303 coeficientes para 48 → 16 kHz y 44 265 para 44.1 → 16 kHz ($\beta = 7.86$). El segundo es largo porque trabaja a $160 \times 44.1\text{ kHz}$, pero la estructura polifasica solo usa $44\,265/160 \approx 277$ coeficientes por muestra de salida.

`ffmpeg` se usa solo para decodificar el formato, siempre en la frecuencia original del archivo: el remuestreo lo hace el proyecto (`decode_with_ffmpeg` y `resample`).

### 2.4. Remuestreo por bloques para el modo en vivo

`StreamingResampler` aplica el mismo filtro sobre un flujo que llega en bloques de unos 250 ms. Mantiene un historial de entrada del tamano del soporte del FIR, remuestrea historial + bloque y emite solo las muestras de salida cuyo soporte quedo completo. El inicio del historial siempre es multiplo de $M$, asi la rejilla de salida no se desplaza entre bloques. El test `test_remuestreo_por_bloques_igual_al_offline` verifica que el resultado coincide con el remuestreo offline de la senal completa con error menor a $10^{-9}$.

### 2.5. Cuantizacion

Los PCM de 16 bits se llevan a punto flotante dividiendo por $2^{15}$ (`scale_pcm`):

$$x[n] = \frac{x_{\text{int16}}[n]}{32768}, \quad x[n] \in [-1, 1)$$

El ruido de cuantizacion de 16 bits tiene una potencia de $\Delta^2/12$ con $\Delta = 2^{-15}$, unos −101 dBFS de valor eficaz: muy por debajo de cualquier silencio real. **No se normaliza el pico**: normalizar cada archivo a escala completa llevaria el ruido de fondo de un silencio al nivel de la voz y borraria la diferencia de energia, que es justamente lo que separa silencio de ruido.

---

## 3. Acondicionamiento (`src/filters.py`)

### 3.1. Por que filtrar

La version anterior solo restaba la media de cada bloque de 3 s. Eso elimina la continua pero no el rumble ni el zumbido de red, y lo hace con un "filtro" cuya respuesta depende del largo del bloque. Ahora hay una cascada disenada con especificaciones:

| Interferencia | Banda | Filtro |
| :--- | :--- | :--- |
| Continua, rumble, viento, golpes al celular | 0–50 Hz | Pasa-altos Butterworth, $f_c = 70\text{ Hz}$, orden 4 |
| Red electrica | 50 Hz ± 0.5 Hz | Notch IIR de 2.º orden, $Q = 10$ |

Ambos se aplican **antes** de extraer caracteristicas y con **el mismo codigo** en entrenamiento, inferencia y modo en vivo (`features.signal_to_windows` es la unica puerta de entrada). Si se filtrara solo en inferencia, el modelo veria en produccion una distribucion distinta a la de entrenamiento.

### 3.2. Pasa-altos Butterworth

El Butterworth de orden $N$ es el filtro **maximamente plano**: las primeras $2N-1$ derivadas de $|H|^2$ son nulas en la banda de paso, asi que no tiene ondulacion. Su respuesta de modulo (prototipo pasa-altos analogico) es

$$|H(j\Omega)|^2 = \frac{1}{1 + \left(\Omega_c / \Omega\right)^{2N}}$$

A $\Omega = \Omega_c$ vale $1/2$ (−3.01 dB) para cualquier $N$, y lejos de $\Omega_c$ cae $20N$ dB/decada ($6N$ dB/octava).

**Eleccion de $f_c = 70\text{ Hz}$:** queda por debajo de la fundamental de la voz masculina mas grave (≈ 85 Hz) y por encima del rumble de manipulacion, viento y climatizacion (< 50 Hz). Mas alto empezaria a recortar $F_0$ de voces graves y el bajo de la musica; mas bajo dejaria pasar viento.

**Eleccion de $N = 4$** (24 dB/octava). Atenuacion de una pasada, medida sobre el diseno:

| $f$ [Hz] | 20 | 50 | 70 | 85 | 100 | 150 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $\|H\|$ [dB] | −43.5 | −12.0 | −3.0 | −0.83 | −0.24 | −0.01 |

Con orden 2 la atenuacion en 20 Hz seria solo ≈ −22 dB; con orden 8 habria mas atenuacion pero los polos quedarian mas cerca del circulo unidad (mas sensibilidad numerica y mas retardo de grupo cerca de $f_c$, con una respuesta al escalon mas oscilante). Orden 4 es el compromiso usual en audio.

**Discretizacion:** transformacion bilineal $s = \frac{2}{T_s}\frac{1 - z^{-1}}{1 + z^{-1}}$, que lleva el semiplano izquierdo al interior del circulo unidad (un analogico estable da un digital estable) pero comprime el eje de frecuencias: $\Omega = \frac{2}{T_s}\tan\left(\frac{\omega}{2}\right)$. `scipy.signal.butter(..., fs=fs)` pre-distorsiona $f_c$ con esa relacion para que el −3 dB caiga exactamente en 70 Hz (verificado en `test_pasa_altos_menos_3db_en_fc`).

**Polos y ceros:** los 4 ceros estan en $z = 1$ (un cero de orden 4 en continua: rechazo total de DC) y los 4 polos en $|z| = 0.975$ y $0.990$, con angulos de ±0.0105 y ±0.0254 rad (`docs/figuras/acond_polos_ceros.png`). Estan tan cerca de $z = 1$ porque $70\text{ Hz}$ es solo el 0.9 % de $f_s$.

### 3.3. Notch de 50 Hz

Un notch de segundo orden pone un par de ceros **sobre** el circulo unidad en $e^{\pm j\omega_0}$, con $\omega_0 = 2\pi f_0 / f_s$, y un par de polos en el mismo angulo con radio $r < 1$:

$$H(z) = b_0 \frac{1 - 2\cos\omega_0 \, z^{-1} + z^{-2}}{1 - 2r\cos\omega_0 \, z^{-1} + r^2 z^{-2}}$$

En $\omega_0$ el numerador se anula (atenuacion infinita en teoria; −274 dB numericamente). Lejos de $\omega_0$, polo y cero estan practicamente a la misma distancia de $e^{j\omega}$ y se compensan: $|H| \approx 1$. El ancho de la muesca a −3 dB es $BW = f_0/Q$ y se relaciona con el radio por $r \approx 1 - \pi BW/f_s$.

**Eleccion de $Q = 10$** ($BW = 5\text{ Hz}$, $r = 0.99902$): la red puede desviarse de 50 Hz; con $Q = 10$ la atenuacion de una pasada en 49.5 y 50.5 Hz sigue siendo de −14 dB (−28 dB con filtfilt). Con $Q = 30$ la muesca seria de 1.7 Hz y en ±0.5 Hz solo atenuaria ≈ −6 dB. La banda que se pierde (47.5–52.5 Hz) ya esta en la zona de rechazo del pasa-altos, asi que el notch no saca nada que el sistema use.

**Por que no los armonicos (100, 150 Hz):** con el mismo $Q$ cada muesca tendria $BW = f_k/Q$, es decir 10 y 15 Hz, justo en el rango de $F_0$ de voces masculinas. El costo para la voz supera la ganancia; queda como opcion en `NOTCH_FREQS_HZ`.

**Por que el notch si hace falta con el pasa-altos:** el pasa-altos solo atenua 12 dB en 50 Hz. Un zumbido fuerte en una grabacion de "silencio" puede estar 20–30 dB por encima del ruido de sala y alcanzar para que el silencio se confunda con ruido. La cascada da −274 dB en 50 Hz y −26 dB en 49.5/50.5 Hz en una sola pasada.

### 3.4. Estabilidad y secciones de segundo orden (SOS)

Un IIR causal es estable (BIBO) si y solo si todos sus polos cumplen $|p_i| < 1$. El radio maximo de la cascada es 0.99902 (`is_stable`, `test_estabilidad`).

Con polos tan cerca de $z = 1$, la forma directa $b/a$ de orden 6 es peligrosa: los coeficientes del denominador quedan muy proximos a los de $(1 - z^{-1})^6$ y un error de redondeo relativo de $10^{-16}$ en ellos puede mover las raices lo suficiente como para cambiar la respuesta o sacar un polo del circulo. Por eso todo el filtrado usa una **cascada de biquads** (`output="sos"`, `tf2sos`): cada seccion de segundo orden tiene sus propios coeficientes y la sensibilidad queda acotada a la de cada par de polos. Son 3 secciones: 2 del pasa-altos y 1 del notch.

### 3.5. Fase, retardo de grupo y `sosfilt` contra `sosfiltfilt`

El retardo de grupo $\tau_g(\omega) = -\frac{d\angle H(e^{j\omega})}{d\omega}$ mide cuanto se demora la envolvente de una componente de frecuencia $\omega$. En una pasada causal, la cascada tiene $\tau_g \approx 8\text{–}9\text{ ms}$ cerca de 50–70 Hz, 4 ms en 100 Hz, 0.8 ms en 200 Hz y practicamente cero por encima de 500 Hz (`docs/figuras/acond_retardo_grupo.png`). Es decir, el filtro no es de fase lineal: los graves se retrasan algunos ms respecto de los agudos.

- **Offline** (`condition_offline`, archivos completos en entrenamiento e inferencia): `sosfiltfilt` filtra hacia adelante y luego la senal invertida en el tiempo. El resultado tiene respuesta $|H(e^{j\omega})|^2$ y **fase nula**: el retardo se cancela exactamente y no hay corrimiento entre la senal y las etiquetas temporales. El costo es que no es causal (necesita la senal entera) y que el modulo queda al cuadrado: todas las atenuaciones en dB se duplican (−87 dB en 20 Hz, −24 dB en 50 Hz sin contar el notch, −0.5 dB en 100 Hz).
- **En vivo** (`CausalConditioner`): no se puede usar el futuro. Se aplica la misma cascada **dos veces hacia adelante** con `sosfilt` y estado interno `zi` conservado entre bloques. Dos pasadas causales dan el mismo modulo $|H|^2$ que filtfilt; solo cambia la fase (retardo doble, unos 17 ms en 70 Hz). Como las caracteristicas usan el modulo del espectro de cada frame, ambos modos producen practicamente los mismos vectores (`test_causal_doble_mismo_modulo_que_offline`, error relativo < $10^{-3}$). Conservar `zi` hace que el resultado no dependa de como se trocee el flujo (`test_causal_independiente_del_troceo`).

---

## 4. Analisis de Corto Plazo (`src/preprocessing.py`, `src/features.py`, `src/fft.py`)

### 4.1. Jerarquia temporal

1. **No estacionariedad global:** una grabacion larga cambia sus propiedades todo el tiempo; una FFT sobre varios segundos mezcla eventos distintos.
2. **Cuasi-estacionariedad local:** en 20–40 ms el tracto vocal y los instrumentos mantienen una configuracion aproximadamente fija. Ahi tiene sentido un espectro.
3. **Contexto de decision:** un frame de 30 ms no alcanza para decidir (una fricativa aislada parece ruido, una pausa entre silabas parece silencio). Se agregan estadisticas de los frames sobre una ventana de varios segundos.

| Nivel | Simbolo | Duracion | Muestras a 16 kHz | Funcion |
| :--- | :---: | :---: | :---: | :--- |
| Ventana de decision | $T_{\text{seg}}$ | 3.0 s | 48 000 | Unidad de clasificacion (media y desvio de los frames) |
| Salto de ventana | $H_{\text{seg}}$ | 0.5 s | 8 000 | Resolucion temporal de la segmentacion |
| Frame de analisis | $N_{\text{win}}$ | 30 ms | 480 | Espectro de corto plazo |
| Salto de frame | $H$ | 10 ms | 160 | Solape de 2/3 entre frames |
| Largo de FFT | $N_{\text{FFT}}$ | — | 512 | Frame + 32 ceros |

**Por que cambiaron los parametros respecto de la version anterior** (512/256): con $H = 10\text{ ms}$ el salto de ventana (0.5 s = 50 frames) y la ventana (3 s = 300 saltos) son multiplos exactos del salto de frame, asi que las ventanas deslizantes se arman **reutilizando** los frames ya calculados de la senal completa (no se recalcula la FFT por cada ventana solapada) y una ventana extraida de un archivo largo es identica a la misma ventana extraida sola. Ademas $N/H = 3$ entero hace que la Hann periodica cumpla COLA (4.3).

El numero de frames de una senal de $L$ muestras es

$$M = 1 + \left\lfloor \frac{L - N_{\text{win}}}{H} \right\rfloor, \qquad M_{3\text{ s}} = 1 + \left\lfloor \frac{48000 - 480}{160} \right\rfloor = 298$$

(`n_frames_for`, `test_conteo_de_frames`).

### 4.2. DFT, simetria hermitica y resolucion

La DFT de un frame de $N$ muestras es

$$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}, \qquad k = 0, \dots, N-1$$

Para $x[n]$ real vale $X[N-k] = X^*[k]$, asi que alcanza con los bins $k = 0, \dots, N/2$: 257 bins para $N_{\text{FFT}} = 512$, en las frecuencias

$$f[k] = k \frac{f_s}{N_{\text{FFT}}} = k \cdot 31.25\text{ Hz}$$

El relleno con ceros (480 → 512) no agrega resolucion real (esa la fija el largo del frame: los lobulos principales de una Hann de 30 ms miden $4 f_s / N_{\text{win}} \approx 133\text{ Hz}$) pero interpola el espectro y permite usar una FFT radix-2.

### 4.3. Ventanas, fuga espectral y COLA

Tomar $N$ muestras equivale a multiplicar por una ventana rectangular; en frecuencia, el espectro se **convoluciona** con el de la ventana (un nucleo de Dirichlet). Si el tono no cae justo en un bin, su energia se reparte en todos los bins: **fuga espectral**. Los lobulos laterales de la rectangular estan a −13 dB y caen 6 dB/octava; los de la Hann estan a −31 dB y caen 18 dB/octava, a cambio de un lobulo principal del doble de ancho.

`scripts/leakage_demo.py` (calculado con la FFT propia, `docs/figuras/fuga_espectral.png`) lo muestra con un tono de 1015.6 Hz (justo entre dos bins) mas otro de 2500 Hz a −50 dB: con la rectangular la fuga del tono fuerte en 2500 Hz llega a −45.5 dB y tapa al debil; con Hann el tono debil se ve en −50.0 dB, su nivel real.

**Ventana de Hann periodica.** Se usa

$$w[n] = 0.5\left(1 - \cos\frac{2\pi n}{N}\right), \qquad 0 \le n \le N-1$$

**Correccion respecto de la version anterior.** La documentacion anterior usaba la Hann *simetrica* (denominador $N - 1$) y afirmaba que con 50 % de solape cumplia COLA. No es exacto: la condicion COLA (*constant overlap-add*)

$$\sum_{m} w[n - mH] = C \quad \forall n$$

la cumple la Hann **periodica** cuando $H = N/R$ con $R \ge 2$ entero, y la simetrica no la cumple para ningun salto (la suma ondula levemente). Con $N = 480$, $H = 160$ ($R = 3$) la suma vale exactamente $C = N/(2H) = 1.5$. Los tests `test_cola_hann_periodica` y `test_hann_simetrica_no_es_cola` lo verifican. Para el analisis (no hay resintesis) COLA no es estrictamente necesaria; lo que garantiza es que todas las muestras de la senal pesan lo mismo en el conjunto de frames, sin zonas sub-representadas entre frames. La Hann simetrica es la adecuada para disenar filtros FIR, no para STFT.

### 4.4. FFT radix-2 propia (`src/fft.py`)

La DFT directa cuesta $O(N^2)$ productos complejos. La FFT de Cooley-Tukey por **decimacion en el tiempo** separa muestras pares e impares:

$$X[k] = E[k] + W_N^k \, O[k], \qquad X[k + N/2] = E[k] - W_N^k \, O[k], \qquad W_N = e^{-j2\pi/N}$$

donde $E$ y $O$ son DFTs de largo $N/2$. Aplicado recursivamente, quedan $\log_2 N$ etapas de $N/2$ "mariposas": $O(N \log_2 N)$. Para $N = 512$, $512^2 = 262\,144$ contra $512 \cdot 9 = 4\,608$ (unas 57 veces menos).

La implementacion es **iterativa**: primero reordena la entrada por inversion de bits (el indice $n = (b_{8} \dots b_0)_2$ va a la posicion $(b_0 \dots b_8)_2$), y luego en cada etapa $m = 2, 4, \dots, N$ combina bloques contiguos de largo $m/2$ con los factores $W_m^k$. Esta vectorizada sobre el primer eje, asi que transforma la matriz de todos los frames de una vez.

Verificacion (`tests/test_fft.py`): coincide con `np.fft.fft` para $N = 2 \dots 4096$ (error < $10^{-9}$), con la DFT directa, cumple Parseval y `rfft_radix2` con relleno de ceros coincide con `np.fft.rfft`. Las 64 caracteristicas calculadas con `FFT_BACKEND = "propia"` coinciden con las de numpy con error relativo < $10^{-8}$ (`test_fft_propia_mismas_caracteristicas`). Para entrenar se usa numpy por velocidad (unas 3 veces mas rapido); el resultado es el mismo.

---

## 5. Caracteristicas (`src/features.py`)

### 5.1. Descriptores por frame (38)

Sea $x_m[n]$ el frame $m$ sin ventana (para RMS y ZCR) y $X_m[k]$ la FFT del frame con Hann; $S_m[k] = |X_m[k]|$ es el espectro de modulo y $P_m[k] = S_m[k]^2$ el de potencia.

**Temporales**

1. **RMS:** $\text{RMS}_m = \sqrt{\frac{1}{N}\sum_n x_m^2[n]}$. Separa silencio de sonido activo y su variacion captura la modulacion silabica.
2. **Tasa de cruces por cero:** fraccion de pares de muestras consecutivas con cambio de signo. Baja en tramos sonoros, alta en fricativas y ruido.

**Espectrales "clasicas"**

3. **Centroide:** $C_m = \sum_k f[k] S_m[k] / \sum_k S_m[k]$, centro de gravedad del espectro ("brillo").
4. **Ancho de banda:** $B_m = \sqrt{\sum_k (f[k] - C_m)^2 S_m[k] / \sum_k S_m[k]}$, dispersion alrededor del centroide.
5. **Energia relativa en 5 subbandas de octava** (0–500, 500–1000, 1000–2000, 2000–4000, 4000–8000 Hz): $R_{m,b} = \sum_{k \in \mathcal{K}_b} P_m[k] / \sum_k P_m[k]$.

**Espectrales nuevas**

6. **Rolloff (85 %):** la menor frecuencia $f_r$ tal que $\sum_{f[k] \le f_r} P_m[k] \ge 0.85 \sum_k P_m[k]$. Separa espectros concentrados en graves (voz sonora) de los de banda ancha (ruido, platillos).
7. **Planitud espectral (Wiener):** cociente entre media geometrica y aritmetica de la potencia,
   $$\text{SFM}_m = \frac{\exp\left(\frac{1}{K}\sum_k \ln P_m[k]\right)}{\frac{1}{K}\sum_k P_m[k]} \in (0, 1]$$
   Vale 1 para ruido blanco y tiende a 0 para un espectro de pocas lineas (armonicos). Es la caracteristica mas directa para "ruido contra tono".
8. **Flujo espectral:** $\Phi_m = \left\| \hat S_m - \hat S_{m-1} \right\|_2$ con $\hat S_m = S_m / \sum_k S_m[k]$. Cuanto cambia el espectro entre frames: alto y muy variable en la voz (transiciones entre fonemas), mas bajo y regular en la musica sostenida y en el ruido estacionario. Normalizar por la suma lo hace independiente del nivel.

**Cepstrales**

9. **13 MFCC** (seccion 5.2).
10. **13 deltas de MFCC** (seccion 5.3).

### 5.2. Banco de filtros mel y MFCC, hechos a mano

La escala mel aproxima la resolucion en frecuencia del oido: lineal hasta ~1 kHz y logaritmica por encima,

$$\text{mel}(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

`mel_filterbank` reparte $40 + 2$ bordes uniformemente en mel entre 20 Hz y 8 kHz, los vuelve a Hz y construye 40 filtros triangulares sobre los 257 bins: el filtro $j$ sube linealmente de $f_{j-1}$ a $f_j$ y baja hasta $f_{j+1}$. Se normalizan por area ($2/(f_{j+1} - f_{j-1})$) para que los filtros anchos de alta frecuencia no dominen por tener mas bins.

Luego:

$$E_m[j] = \sum_k P_m[k] \, H_j[k], \qquad c_m[i] = \sum_{j=0}^{39} \ln\left(E_m[j] + 10^{-10}\right) \, \alpha_i \cos\left(\frac{\pi i (2j + 1)}{80}\right), \quad i = 0, \dots, 12$$

con la DCT-II ortonormal ($\alpha_0 = \sqrt{1/40}$, $\alpha_i = \sqrt{2/40}$), tambien construida a mano (`dct_matrix`, `test_dct_ortonormal`). Por que funciona:

- El **logaritmo** convierte el producto fuente × filtro (excitacion glotal por tracto vocal) en una suma, y comprime el rango dinamico como el oido.
- La **DCT** decorrela las energias de bandas vecinas (que estan muy correlacionadas) y concentra la envolvente espectral suave en los primeros coeficientes. $c_0$ es proporcional a la energia logaritmica total; $c_1, c_2, \dots$ describen la forma de la envolvente (formantes). Quedarse con 13 descarta la estructura fina (los armonicos de $F_0$).

Los MFCC son la caracteristica estandar en reconocimiento de habla y de clases de audio porque resumen la envolvente espectral en pocos numeros casi independientes.

### 5.3. Deltas

La dinamica temporal de los MFCC se estima con una regresion lineal sobre $\pm D$ frames ($D = 2$, ±20 ms):

$$\Delta c_m = \frac{\sum_{d=1}^{D} d \, (c_{m+d} - c_{m-d})}{2 \sum_{d=1}^{D} d^2}$$

(bordes replicados). En la voz los MFCC cambian rapido (coarticulacion) y el desvio de los deltas es alto; en la musica sostenida y el ruido estacionario es bajo.

### 5.4. Agregacion en la ventana de 3 s (64 caracteristicas)

Sobre los 298 frames de cada ventana:

| Grupo | Contenido | Dimension |
| :--- | :--- | :---: |
| `base` | media y desvio de RMS, ZCR, centroide, ancho de banda, 5 subbandas (las 18 de la version anterior) | 18 |
| `espectrales` | media y desvio de rolloff, planitud y flujo | 6 |
| `mfcc` | media y desvio de 13 MFCC | 26 |
| `deltas` | desvio de 13 deltas (la media de un delta es ≈ 0) | 13 |
| `dinamica` | fraccion de frames de baja energia: $\frac{1}{M}\#\{m : \text{RMS}_m < 0.5 \, \overline{\text{RMS}}\}$ | 1 |

La **fraccion de baja energia** es una medida clasica para voz contra musica: el habla tiene muchas pausas cortas entre palabras, la musica casi ninguna.

Media y desvio capturan el comportamiento tipico y su variabilidad: la voz se caracteriza mas por la **variacion** (desvio de ZCR, de flujo, de deltas) que por el valor medio.

### 5.5. Una sola cadena para entrenamiento e inferencia

`signal_to_windows(x, hop)` hace siempre: acondicionar → frames → descriptores → ventanas. El dataset la llama con `hop = 3 s` (sin solape), el segmentador con `hop = 0.5 s` y el modo en vivo sobre los ultimos 3 s ya acondicionados en forma causal. El test `test_equivalencia_entrenamiento_inferencia` verifica que el camino del dataset (`process_file`) y el de inferencia dan exactamente los mismos vectores para el mismo audio.

Las senales de menos de 3 s se repiten periodicamente hasta completar la ventana: rellenar con ceros agregaria silencio artificial y sesgaria la clasificacion.

---

## 6. Construccion del Dataset (`src/dataset.py`, `src/augment.py`)

### 6.1. MUSAN

MUSAN (OpenSLR 17) trae unas 109 h a 16 kHz: musica (≈ 42 h, varios generos, con y sin voz), habla (≈ 60 h, LibriVox y discursos) y ruido (≈ 6 h, sonidos de Freesound y otros). No trae silencio. El ruido es la clase con menos audio, y en la version anterior era la peor clasificada (recall de 0.66).

### 6.2. Particion sin fuga de informacion

Si se cortan los archivos en ventanas y despues se reparten al azar, ventanas contiguas del mismo locutor, cancion o fuente de ruido quedan en entrenamiento y en validacion: el modelo memoriza el timbre del canal o del locutor y las metricas salen optimistas (*data leakage*).

1. Cada archivo se asigna a uno de **5 pliegues**, estratificado por clase y equilibrando la cantidad de audio por pliegue (`assign_folds`). En musica, los ANNOTATIONS de MUSAN traen el artista: todos los archivos de un mismo artista van al mismo pliegue, aunque esten en subconjuntos distintos (`read_source_groups`). En habla los ANNOTATIONS solo traen sexo e idioma, no el locutor, asi que cada archivo es su propio grupo; es una limitacion conocida (un lector de LibriVox con varios capitulos podria quedar en dos pliegues).
2. Todo el audio sintetico de un pliegue (silencio, mezclas, canal de celular) se fabrica **solo con recortes de archivos de ese pliegue**.
3. Los pliegues 0–3 se usan para validacion cruzada y seleccion de modelo; el **pliegue 4 queda reservado** como prueba final y como fuente del banco de recortes para las curvas de robustez.

### 6.3. Clase silencio

En una grabacion real el silencio absoluto no existe; hay ruido de fondo de muy bajo nivel. Se define silencio como **senal sin eventos con nivel RMS entre −75 y −36 dBFS medido sobre la senal acondicionada** (`SILENCE_DBFS_RANGE`), y se generan 9 tipos por igual (`make_silence`).

**Como se llego a esta definicion (y por que importa medir despues del filtro).** La primera version fijaba el nivel del silencio sintetico entre −85 y −52 dBFS *antes* del acondicionamiento. Las grabaciones de celular de `data/real/` (notas de voz Opus a 48 kHz) mostraron dos problemas:

1. El silencio de una pieza grabado con el celular queda en **−40 a −42 dBFS** ya acondicionado: el control automatico de ganancia (AGC) sube el ruido de fondo cuando no hay senal. Esa grabacion se clasificaba 100 % como ruido; atenuada 15 dB pasaba a silencio (0.93).
2. Subir solo el techo nominal no alcanzo: como el nivel se fijaba antes del filtro, el ruido marron y el rosa (con mucha energia debajo de 70 Hz) quedaban, despues del pasa-altos, muy por debajo de lo nominal; el 90 % del silencio de entrenamiento terminaba por debajo de unos −49 dBFS efectivos. Ademas el silencio con AGC tiene una forma que ningun tipo sintetico cubria: 81 % de la energia por debajo de 500 Hz, planitud muy baja (0.005) y flujo espectral alto (0.09) porque el AGC "bombea" el nivel.

La correccion fue fijar el nivel sobre la senal ya acondicionada (la que ve el modelo, coherente con como se midio la grabacion) y agregar el tipo `agc_celular`. Como ese tipo se diseno despues de ver la grabacion, **esa grabacion de silencio ya no es una prueba independiente**; una validacion limpia requiere otro silencio grabado en otro lugar.

| Tipo | Como se genera | Que representa |
| :--- | :--- | :--- |
| `sala` | ruido real de MUSAN del pliegue, pasa-bajos de 1.5–5 kHz | tono de sala, paredes que absorben agudos |
| `rosa` | espectro $\propto 1/f$ | ruido ambiente generico |
| `marron` | espectro $\propto 1/f^2$ | rumble de edificio, trafico lejano |
| `siseo` | blanco con pasa-altos de 1.er orden en 2 kHz | ruido termico del preamplificador |
| `dither` | ceros digitales + dither TPDF de ±1 LSB, cuantizado a 16 bits | silencio digital de una app o un archivo |
| `pausas_voz` | frames del 10 % de menor energia de los archivos de voz | pausas reales entre frases |
| `zumbido` | 50 Hz + armonicos 2, 3, 5 con desvio de ±0.3 Hz, mas ruido rosa | silencio con equipos enchufados |
| `muy_bajo` | ruido de color aleatorio a −85/−70 dBFS | placa de sonido en una pieza muy silenciosa |
| `agc_celular` | ruido de pendiente $1/f^{1.5\text{–}2.5}$, pasa-bajos de 0.4–1.5 kHz y modulacion lenta de nivel (0.3–2 Hz, 2–8 dB) | pieza en silencio grabada con un celular con AGC |

Los ruidos de color se generan en frecuencia (espectro blanco complejo multiplicado por $f^{-\alpha/2}$, con la pendiente saturada por debajo de 20 Hz) y todos se cuantizan a 16 bits como lo haria el conversor. Una sexta parte del silencio pasa ademas por el canal de celular.

La frontera de nivel tiene consecuencias en las clases activas:

- En **ruido** se descartan las ventanas de MUSAN por debajo de −34 dBFS (`NOISE_MIN_DBFS`): son ambientes casi vacios que, con esta definicion, son ruido de fondo, es decir silencio. Entre −36 y −34 dBFS queda una franja de 2 dB sin ejemplos para que las dos definiciones no se pisen.
- En **voz y musica** el piso queda en −50 dBFS (`ACTIVE_MIN_DBFS`; se descartan colas de pistas y pausas largas). Aunque su rango de nivel se superpone con el del silencio, las separa la forma del espectro (armonicos, modulacion, variabilidad), no el nivel.

### 6.4. Augmentation

Cada clase tiene el mismo presupuesto por pliegue (`PER_CLASS_PER_FOLD = 3000` ventanas): hasta el 60 % son ventanas limpias de MUSAN y el resto se completa con datos sinteticos (1/3 mezclas, 1/6 reverb + ganancia, 1/2 canal de celular). Asi la clase ruido, la mas escasa, queda con el mismo peso que las otras. Ademas cada archivo recibe una ganancia aleatoria de −10 a +6 dB.

**Mezcla a SNR controlada** (`mix_at_snr`): $y = s + a \, r$ con $a$ tal que

$$\text{SNR} = 10\log_{10}\frac{P_s}{a^2 P_r}, \qquad a = \sqrt{\frac{P_s}{P_r \, 10^{\text{SNR}/10}}}$$

- **Voz + ruido** y **musica + ruido** con SNR uniforme en 0–20 dB, etiquetadas con la clase de primer plano. Son negativos dificiles para la clase ruido: el modelo tiene que aprender que voz con ventilador de fondo sigue siendo voz.
- **Ruido con fondo** de voz o musica 8–20 dB por debajo, etiquetado ruido: negativos dificiles para voz y musica (por ejemplo, un bar).

**Reverberacion sintetica** (`synthetic_reverb`): respuesta al impulso $h[n] = \delta[n] + g \, e[n] \, e^{-6.91 n/(T_{60} f_s)}$ con $e[n]$ ruido blanco y $T_{60} \in [0.15, 0.7]\text{ s}$. El factor 6.91 sale de $e^{-6.91} = 10^{-3}$: la energia cae 60 dB en $T_{60}$ segundos. Las primeras reflexiones empiezan a los 3 ms.

### 6.5. Simulacion del canal de celular (`phone_channel`)

Por recorte:

1. Reverb de sala con probabilidad 0.6.
2. **Pasa-banda** Butterworth de orden 4: banda angosta 300–3400 Hz (telefonia clasica, 40 % de los casos) o banda ancha 100–7000 Hz (notas de voz, mensajeria).
3. **Ganancia** de −12 a +6 dB y, en el 15 % de los casos, **saturacion suave** $\tanh(d \, x)/d$ que imita un AGC agresivo.
4. **Codec con perdida** real, codificando y decodificando con ffmpeg: AMR-NB a 12.2 kbit/s (a 8 kHz, el codec de las llamadas GSM/3G) para la banda angosta; MP3 32/64 kbit/s, AAC 32/64 kbit/s u Opus 16 kbit/s para la banda ancha; una fraccion queda sin codec. Los recortes se procesan por lotes, concatenados con 0.25 s de silencio entre ellos para que el solape de tramas del codec no mezcle recortes vecinos, y se realinean estimando el retardo del codec por correlacion cruzada (`estimate_lag`).

Los codecs perceptuales eliminan componentes enmascaradas y cuantizan las bandas de alta frecuencia: cambian la planitud, el rolloff y los MFCC altos. Entrenar con ellos evita que el modelo dependa de detalles que una grabacion de celular no conserva.

---

## 7. Modelos y Validacion (`src/train.py`)

### 7.1. Estandarizacion

Las 64 caracteristicas tienen escalas muy distintas (RMS en $[0, 1]$, centroide en miles de Hz, MFCC del orden de decenas). Cada modelo es un `Pipeline` que empieza con `StandardScaler`:

$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

con $\mu, \sigma$ estimadas **solo con los datos de entrenamiento de cada pliegue**. Si se estimaran con todo el dataset, informacion de validacion se filtraria al preprocesamiento.

### 7.2. Clasificadores comparados

| Modelo | Idea | Configuracion |
| :--- | :--- | :--- |
| k-NN | clase mayoritaria (ponderada por $1/d$) entre los $k$ vecinos mas cercanos en distancia euclidea | $k = 15$ |
| MLP | red de 2 capas ocultas ReLU, salida softmax, entrenada con Adam y parada temprana | 128–64, $\alpha = 10^{-3}$ |
| SVM | maximo margen con nucleo gaussiano $K(x, x') = e^{-\gamma\|x - x'\|^2}$; probabilidades por escalado de Platt | $C = 10$, $\gamma$ = `scale` |
| Random Forest | promedio de 300 arboles entrenados con bootstrap y subconjuntos aleatorios de caracteristicas | hojas de al menos 2 |
| Gradient Boosting | suma de arboles chicos, cada uno ajustado al gradiente de la log-perdida del anterior; version por histogramas | 400 iteraciones, $\eta = 0.08$ |

Todos son "tradicionales", como pide la consigna: no hay redes profundas ni modelos preentrenados. Todos dan `predict_proba`, que necesita la segmentacion (seccion 8).

### 7.3. Desbalance y foco en ruido

El dataset se arma balanceado por clase (seccion 6.4). Ademas SVM, Random Forest y Gradient Boosting usan `class_weight="balanced"`, que pondera cada muestra por $n / (K \cdot n_c)$: si quedara un desbalance residual, una clase con menos muestras pesa lo mismo en la funcion de costo. k-NN y MLP no aceptan pesos, pero reciben datos balanceados. Se informa siempre el **recall de ruido**, que era el punto debil de la version anterior.

### 7.4. Validacion cruzada agrupada y seleccion

Se entrena cada modelo 4 veces dejando afuera uno de los pliegues 0–3 (que son disjuntos en archivos y fuentes) y se promedian las metricas. Se elige por **F1 macro**,

$$F_1^{\text{macro}} = \frac{1}{K}\sum_{c=1}^{K} \frac{2 \, P_c R_c}{P_c + R_c}$$

porque, a diferencia de la accuracy, cada clase pesa lo mismo: un modelo que ignora el ruido no puede compensarlo acertando mucha voz. El modelo elegido se reentrena con los pliegues 0–3 y se evalua **una sola vez** en el pliegue 4. Ese mismo modelo es el que se guarda en `models/modelo.joblib`, asi las metricas informadas corresponden exactamente al modelo que usa la interfaz.

### 7.5. Ablation de caracteristicas

Con el modelo elegido se repite la validacion cruzada agregando grupos de caracteristicas de a uno (`base` → `+espectrales` → `+mfcc` → `+deltas` → `+dinamica`) y con solo `mfcc + deltas`. Mide cuanto aporta cada grupo y justifica que la dimension 64 no es arbitraria. Resultados en `models/metricas.json` y `docs/figuras/ablation.png`.

---

## 8. Segmentacion (`src/segmenter.py`, `src/live.py`)

### 8.1. Ventana deslizante con solape

La version anterior clasificaba bloques fijos de 3 s: la resolucion temporal era de 3 s y los ultimos segundos que no completaban un bloque quedaban sin clasificar. Ahora se toma una ventana de 3 s **cada 0.5 s** (83 % de solape) y el modelo da un vector de probabilidades $\mathbf{p}_i \in \mathbb{R}^4$ por ventana.

**Cola final:** si el audio no termina justo en la rejilla de 0.5 s, se agrega una ventana final alineada al ultimo frame (`window_starts(include_tail=True)`). Ningun tramo queda sin clasificar.

### 8.2. De ventanas a celdas

El eje temporal se divide en **celdas** de 0.5 s. La probabilidad de la celda $t$ es el promedio de las ventanas que contienen su centro (hasta 6):

$$\mathbf{q}_t = \frac{1}{|\mathcal{W}_t|} \sum_{i \in \mathcal{W}_t} \mathbf{p}_i$$

Promediar ventanas solapadas ya suaviza; ademas cada celda recibe evidencia de los 2.5 s anteriores y posteriores. La contracara es que un cambio de clase se "difumina" sobre unos ±1.5 s: la precision de un borde esta limitada por el largo de la ventana, no por el salto.

### 8.3. Suavizado con mediana

`smooth_median` aplica un filtro de mediana de 5 celdas (2.5 s) a cada componente de $\mathbf{q}_t$ y renormaliza. La mediana elimina picos aislados sin desplazar los bordes (a diferencia de un promedio movil, que los inclina). Es simple pero decide celda por celda, sin un modelo de cuanto deberia durar un segmento.

### 8.4. Suavizado con Viterbi

Se modela la secuencia de clases como una cadena de Markov oculta: el estado $s_t$ es la clase de la celda, la "observacion" es la salida del clasificador. Se busca

$$\hat s_{1:T} = \arg\max_{s_{1:T}} \sum_{t=1}^{T} \ln q_t(s_t) \; - \; \lambda \sum_{t=2}^{T} \mathbb{1}[s_t \ne s_{t-1}]$$

Es un HMM con matriz de transicion $P(s_t = j \mid s_{t-1} = i) = 1 - \varepsilon$ si $i = j$ y $\varepsilon/(K-1)$ si no, usando las probabilidades a posteriori del clasificador como verosimilitudes escaladas (con prior uniforme, porque el entrenamiento es balanceado). La penalidad por cambio es

$$\lambda = \ln\frac{(1 - \varepsilon)(K - 1)}{\varepsilon} \quad \Longleftrightarrow \quad \varepsilon = \frac{K - 1}{e^{\lambda} + K - 1}$$

El maximo se calcula exactamente con **programacion dinamica** (algoritmo de Viterbi) en $O(T K^2)$: para cada celda y estado se guarda el mejor puntaje acumulado y de que estado se vino, y al final se recorre el camino hacia atras.

**Interpretacion de $\lambda = 2$ nats** (`TRANSITION_PENALTY`): un cambio de clase se acepta si la nueva clase es, acumulado sobre el tramo, al menos $e^2 \approx 7.4$ veces mas probable que seguir en la anterior. Un pico aislado tiene que pagar dos cambios (entrar y salir, $2\lambda = 4$ nats): una celda con $q = 0.7$ contra $0.1$ aporta $\ln 7 \approx 1.95$ y dos celdas $3.9$, que no alcanzan; tres celdas seguidas si. La duracion minima de 1.5 s (3 celdas) completa el criterio. $\lambda = 0$ reproduce el argmax celda a celda (`test_viterbi_sin_penalidad_es_argmax`). La primera version usaba $\lambda = 6$; con la definicion final de silencio (seccion 6.3) esa penalidad absorbia demasiados tramos reales y se bajo a 2 (seccion 9.3).

### 8.5. Duracion minima y fusion

Despues del suavizado, cualquier corrida mas corta que `MIN_SEGMENT_DURATION = 1.5 s` se absorbe en el vecino (izquierdo o derecho) cuya clase tiene mayor probabilidad media en esa corrida, empezando por la mas corta (`enforce_min_duration`). Las celdas contiguas de la misma clase se fusionan en un intervalo, y su **confianza** es la probabilidad media (antes del suavizado) de la clase asignada en ese intervalo. Tambien se devuelve el vector medio de las cuatro probabilidades.

### 8.6. Modo en vivo: Viterbi causal

En vivo no se conoce el futuro, asi que no se puede hacer el recorrido hacia atras de Viterbi. Se usa el **filtrado hacia adelante** del mismo HMM:

$$\boldsymbol{\alpha}_t \propto \mathbf{p}_t \odot \left(T^{\top} \boldsymbol{\alpha}_{t-1}\right)$$

con $T$ la misma matriz de transicion elevada a la cantidad de celdas de 0.5 s que dura un paso de decision (1 s → $T^2$). Cada segundo se clasifica la ultima ventana de 3 s ya acondicionada en forma causal; la etiqueta mostrada es $\arg\max \boldsymbol{\alpha}_t$. Un cambio sostenido se confirma en 1–2 decisiones; un pico aislado no. La latencia total es la de la ventana (la decision describe los ultimos 3 s) mas el retardo de grupo del filtro y del remuestreador (ms).

---

## 9. Evaluacion

### 9.1. Metricas

- **Por ventana** (`src/train.py`): accuracy, F1 por clase y macro, recall por clase y matriz de confusion, en validacion cruzada y en el pliegue de prueba. La prueba se desglosa por tipo de muestra (limpia, mezcla, reverb, cada codec, cada tipo de silencio).
- **Por celda de 0.5 s** (`scripts/eval_real.py` y secuencias sinteticas): proporcion de celdas con la clase correcta, cantidad de segmentos predichos contra reales y error de ubicacion de los bordes.

### 9.2. Curvas de robustez (`scripts/eval_robustness.py`)

Sobre 300 recortes por clase del pliegue de prueba, comparando el sistema nuevo con la version anterior reproducida exactamente (`scripts/legacy_pipeline.py`):

- **Accuracy contra SNR:** voz y musica mezcladas con ruido del pliegue de prueba a 20, 10, 5, 0 y −5 dB.
- **Accuracy contra frecuencia de muestreo:** cada recorte se lleva a 44.1, 11.025, 8 y 6 kHz y se vuelve a 16 kHz con anti-alias (el contenido queda limitado a $f_s/2$, como en una grabacion a esa tasa), y un caso a 8 kHz **sin** anti-alias para medir el efecto del aliasing.
- **Accuracy contra codec:** MP3 128 y 32 kbit/s, AAC 32 kbit/s, Opus 16 kbit/s, AMR-NB y el canal de celular completo.
- **Segmentacion:** 30 secuencias de 6–10 tramos de 3–15 s con clases alternadas, armadas con recortes del pliegue de prueba, segmentadas sin suavizado, con mediana y con Viterbi para $\lambda \in \{2, 4, 6, 10\}$.

### 9.3. Resultados medidos

Todos los valores salen de `models/metricas.json`, `docs/resultados/robustez.json` y `docs/resultados/eval_real.json`, con la definicion final de silencio (seccion 6.3).

**Validacion cruzada (pliegues 0–3, 48 000 ventanas) y prueba (pliegue 4, 12 000 ventanas):**

| Modelo | F1 macro CV | Recall ruido CV |
| :--- | :---: | :---: |
| k-NN | 0.881 ± 0.005 | 0.741 |
| Random Forest | 0.919 ± 0.012 | 0.852 |
| MLP | 0.934 ± 0.007 | 0.865 |
| SVM | 0.938 ± 0.012 | 0.876 |
| **Gradient Boosting** (elegido) | **0.944 ± 0.011** | **0.890** |

En el pliegue de prueba, Gradient Boosting da accuracy 0.943 y F1 macro 0.942, con recall de 0.938 (musica), 0.974 (voz), 0.871 (ruido) y 0.988 (silencio). La confusion dominante es ruido → musica (237 de 3000), seguida de ruido → silencio (99), que es la frontera de nivel de la seccion 6.3. El tipo de muestra mas dificil es ruido con voz o musica de fondo (0.767). GB supera al SVM por 0.006, del orden de un desvio.

**Ablation (Gradient Boosting, CV):** base (18) 0.881 → +espectrales (24) 0.924 → +MFCC (50) 0.938 → +deltas (63) 0.940 → +baja energia (64) 0.944. Solo MFCC + deltas (39) da 0.912: las caracteristicas de la consigna aportan informacion que los MFCC no tienen. El salto mas grande lo dan planitud, rolloff y flujo (+0.044), que es tambien la mayor mejora del recall de ruido (0.778 → 0.865).

**Robustez, nuevo contra anterior** (accuracy; SNR sobre voz y musica, el resto sobre las 4 clases, mismos recortes para los dos sistemas):

| Condicion | Nuevo | Anterior |
| :--- | :---: | :---: |
| SNR 20 / 10 / 5 / 0 dB | 0.955 / 0.928 / 0.823 / 0.667 | 0.893 / 0.732 / 0.605 / 0.538 |
| fs 16 kHz / 11.025 / 8 / 6 kHz | 0.961 / 0.934 / 0.905 / 0.871 | 0.873 / 0.766 / 0.654 / 0.644 |
| 8 kHz sin anti-alias | 0.891 | 0.673 |
| MP3 32 kbit/s / AAC 32 / Opus 16 | 0.959 / 0.954 / 0.948 | 0.867 / 0.874 / 0.873 |
| AMR-NB | 0.871 | 0.663 |
| Canal de celular completo | 0.939 | 0.742 |

La mejora mas grande esta donde la version anterior no tenia datos de entrenamiento: banda angosta y AMR-NB (con AMR-NB el recall de ruido del sistema anterior cae a 0.24; el nuevo queda en 0.85).

**Sobre el anti-alias, honestamente:** a 8 kHz la diferencia con y sin anti-alias es de 1.4 puntos (0.905 contra 0.891), del orden del error de medicion con 1200 recortes (≈ 0.009). En MUSAN hay poca energia entre 4 y 8 kHz y lo que se pliega es debil. El anti-alias se justifica por fidelidad de la senal (el alias de un tono pasa de −3 dB a −100 dB, seccion 2.2) y porque no cuesta nada, mas que por la ganancia de accuracy medida en este conjunto.

**Segmentacion** (30 secuencias continuas del pliegue de prueba, 244 tramos reales):

| Variante | Acierto por celda | Segmentos predichos | Error de borde mediano / medio |
| :--- | :---: | :---: | :---: |
| Sin suavizado | 0.870 | 282 | 0.86 s / 0.98 s |
| Mediana | 0.874 | 265 | 0.89 s / 0.99 s |
| **Viterbi λ = 2** (por defecto) | **0.876** | 240 | 0.91 s / 1.20 s |
| Viterbi λ = 4 | 0.873 | 218 | 0.95 s / 1.64 s |
| Viterbi λ = 6 | 0.872 | 212 | 0.96 s / 1.73 s |
| Viterbi λ = 10 | 0.867 | 206 | 0.98 s / 1.91 s |

Con λ = 2 el numero de segmentos queda casi igual al real (240 contra 244) y el acierto por celda es el mejor. Penalidades mas altas absorben tramos reales cortos: menos segmentos que los reales y error medio de borde mayor. En la primera version del silencio (techo de −52 dBFS) el optimo habia sido λ = 6 (0.899); con el silencio redefinido hay mas celdas ambiguas entre ruido y silencio y una penalidad alta las arrastra. **Advertencia:** λ = 2 se eligio mirando estas secuencias, que salen del pliegue de prueba, asi que su fila es algo optimista; las demas filas describen la sensibilidad.

### 9.4. Grabaciones reales (`scripts/eval_real.py`)

MUSAN no tiene grabaciones de celular. `data/real/` recibe grabaciones propias (instrucciones en `data/real/LEEME.md`); las etiquetas salen del nombre del archivo o de un CSV con intervalos. Se compara el sistema completo, el mismo sin suavizado y la version anterior, por celdas de 0.5 s.

Con cuatro notas de voz de celular (Opus, 48 kHz, 20–25 s cada una):

| Archivo | Resultado | Acierto por celda |
| :--- | :--- | :---: |
| `voz_01` | voz todo el archivo | 1.00 |
| `ruido_01` | ruido todo el archivo | 1.00 |
| `musica_01` | voz 0–8 s, musica 8–20 s | 0.61 |
| `silencio_01` | ruido todo el archivo | 0.00 |

Total: 0.678 por celda (0.655 sin suavizado; version anterior 0.379).

El **silencio** de un celular con control automatico de ganancia queda en −41 dBFS y con una forma espectral que varia en el tiempo (el desvio del primer MFCC es el triple que en los silencios sinteticos). Con eso es indistinguible de un ruido ambiente bajo: el AGC le quita significado al nivel absoluto, que es justamente lo que separa silencio de ruido. Se probaron tres correcciones (subir el techo de nivel, fijarlo despues del filtro, agregar un silencio tipo AGC y uno armado con ambientes reales de MUSAN): la confianza en "ruido" bajo de 0.999 a 0.57, pero no cambio la decision, y seguir habria sido ajustar el modelo a un solo archivo. Queda como limitacion documentada. Soluciones posibles: grabar con una app sin AGC, o detectar silencio en forma relativa (tramos mucho mas bajos que el resto de la misma grabacion), lo que requiere una referencia dentro del archivo.

La **musica** sale como voz en los primeros 8 s. Es consistente con la confusion mas dificil del sistema: musica con voz cantada y poco acompanamiento. Con una sola grabacion no se puede distinguir entre esa explicacion y un error del modelo.

---

## 10. Interfaz y Servidor (`src/server.py`, `web/`)

Servidor HTTP de la biblioteca estandar, sin dependencias externas ni recursos de internet (la interfaz funciona offline). Los graficos estan dibujados con canvas en `web/app.js`. Pestanas:

- **Archivo:** carga, parametros (salto, suavizado, penalidad, duracion minima), forma de onda, segmentos, probabilidad por celda, tabla con confianza, exportacion JSON/CSV/SRT. Al hacer clic en un segmento, el panel *por que* muestra forma de onda, espectrograma de la senal acondicionada, curvas de RMS, ZCR, centroide y planitud, probabilidades, la posicion del segmento en un PCA 2D del espacio de caracteristicas con sus 10 vecinos de entrenamiento mas cercanos, y las caracteristicas que mas lo separan de la segunda clase mas probable (en $z$-scores, comparadas con la media de cada clase).
- **En vivo:** microfono (sin cancelacion de eco, supresion de ruido ni AGC del navegador), espectrograma en tiempo real y etiqueta cada 1 s con filtrado HMM. Transporte por POST de bloques de ~250 ms; no hace falta WebSocket.
- **Acondicionamiento:** Bode, retardo de grupo, polos y ceros, respuesta del anti-alias y comparacion A/B de la senal antes y despues del filtro.
- **Aliasing:** tono o barrido llevado de 48 kHz a 6–16 kHz con y sin anti-alias; espectros y audio.
- **Metricas:** validacion cruzada, prueba, matriz de confusion, ablation, robustez y grabaciones reales.

---

## 11. Cambios Respecto de la Version Anterior

| Aspecto | Antes | Ahora | Por que |
| :--- | :--- | :--- | :--- |
| Acondicionamiento | resta de la media por bloque | pasa-altos 70 Hz + notch 50 Hz, SOS, fase nula | rumble y zumbido de red tambien distorsionan las caracteristicas |
| Remuestreo | ffmpeg, filtro implicito | `resample_poly` con FIR Kaiser de 80 dB | anti-alias controlado y documentado |
| Frames | 512/256, Hann simetrica | 480/160, Hann periodica, FFT 512 | COLA exacta; rejilla compatible con la ventana deslizante |
| Caracteristicas | 18 | 64 (MFCC, deltas, rolloff, planitud, flujo, baja energia) | ver ablation |
| Silencio | pausas de voz y ruido atenuado | 9 tipos con nivel definido tras el filtro, incluido el de un celular con AGC | el silencio real de un celular es ruido de sala, siseo, ceros con dither o ruido subido por el AGC |
| Augmentation | ninguna | mezclas a SNR, reverb, canal de celular con codecs reales | robustez frente a lo que no se puede filtrar |
| Validacion | una particion 80/20 por archivo | 5 pliegues por archivo y fuente, CV + prueba | estimacion con varianza y sin fuga |
| Modelos | k-NN y MLP | k-NN, MLP, SVM, RF, GB | comparacion mas completa |
| Segmentacion | bloques de 3 s, cola descartada | ventana de 3 s cada 0.5 s, Viterbi, duracion minima, cola incluida | resolucion de 0.5 s y segmentos estables |

---

## 12. Preguntas Probables en la Defensa

**¿Por que 16 kHz y no 44.1 kHz?**
Con 16 kHz la Nyquist es 8 kHz: entran los formantes de la voz, las fricativas y casi todo lo que distingue musica de ruido. 44.1 kHz cuesta 2.75 veces mas y no aporta para estas cuatro clases. MUSAN ademas esta a 16 kHz.

**¿Que pasa si se submuestrea sin filtro?**
Lo que esta por encima de la nueva Nyquist se pliega a $|f - k f_s|$ y aparece como una componente que no existia. No se puede deshacer despues. La pestana Aliasing lo muestra: 5 kHz a $f_s = 8\text{ kHz}$ aparece en 3 kHz casi con el mismo nivel; con el FIR anti-alias queda en −100 dB. En la clasificacion de MUSAN a 8 kHz la diferencia fue chica (0.905 con filtro, 0.891 sin), del orden del error de medicion, porque hay poca energia entre 4 y 8 kHz; el filtro se justifica sobre todo por fidelidad.

**¿Como disenaron el anti-alias?**
FIR de fase lineal con ventana de Kaiser: rechazo de 80 dB desde la nueva Nyquist (8 kHz) y transicion de 7.2 a 8 kHz. Kaiser da el orden minimo para esa atenuacion y transicion: 303 coeficientes de 48 a 16 kHz. Se le pasa a `resample_poly`, que lo aplica en forma polifasica.

**¿Por que un pasa-altos en 70 Hz y de orden 4?**
70 Hz esta debajo de la fundamental de voz mas grave (~85 Hz) y arriba del rumble y el viento. Orden 4 da 24 dB/octava: −43 dB en 20 Hz y solo −0.24 dB en 100 Hz. Mas orden pone los polos mas cerca del circulo unidad y aumenta el retardo de grupo cerca del corte sin una ganancia que importe.

**¿Por que Butterworth?**
Es maximamente plano en la banda de paso: no tiene ondulacion, asi que no altera la energia relativa entre bandas que usan las caracteristicas. Un Chebyshev o eliptico cortaria mas abrupto pero con ondulacion en banda de paso o rechazo.

**¿Por que el notch tiene Q = 10?**
$BW = f_0/Q = 5\text{ Hz}$. La red no esta exactamente en 50 Hz; con Q = 10 sigue atenuando 14 dB en ±0.5 Hz (28 dB con filtfilt). Con Q mas alto la muesca seria tan angosta que un pequeno desvio la esquivaria. La banda perdida ya esta en el rechazo del pasa-altos.

**¿El notch no es redundante con el pasa-altos?**
No: el pasa-altos solo atenua 12 dB en 50 Hz. Un zumbido fuerte en un silencio quedaria por encima del ruido de sala y lo haria parecer ruido. El notch pone un cero exactamente sobre la circunferencia en 50 Hz.

**¿Como se sabe que el filtro es estable?**
Todos los polos tienen modulo menor a 1 (el maximo es 0.99902). La bilineal lleva un analogico estable a un digital estable. Y se implementa en secciones de segundo orden para que el redondeo de coeficientes no mueva los polos, que con fc/fs = 0.4 % estan muy cerca de z = 1.

**¿Que es el retardo de grupo y por que importa?**
Es la derivada de la fase respecto de la frecuencia cambiada de signo: cuanto se demora la envolvente de cada componente. El IIR causal retrasa los graves unos 8 ms cerca de 70 Hz y casi nada arriba de 500 Hz. Offline se cancela con filtfilt (fase nula). En vivo queda, pero son milisegundos frente a una ventana de 3 s.

**¿Diferencia entre `sosfilt` y `sosfiltfilt`?**
`sosfilt` es causal: modulo $|H|$, fase no nula. `sosfiltfilt` filtra ida y vuelta: modulo $|H|^2$ y fase nula, pero necesita la senal completa. En vivo se aplica la cascada dos veces hacia adelante para tener el mismo $|H|^2$ que offline y que el modelo vea el mismo tipo de senal.

**¿Por que se filtra tambien al entrenar?**
Porque el filtro forma parte del sistema de medida. Si el modelo se entrena sin filtrar y en inferencia se filtra, ve otra distribucion de caracteristicas. Hay una sola funcion (`signal_to_windows`) para las tres etapas y un test que verifica que dan los mismos vectores.

**¿Por que frames de 30 ms con Hann?**
En 20–40 ms la voz es cuasi-estacionaria. La ventana de Hann reduce la fuga espectral: lobulos laterales de −31 dB contra −13 dB de la rectangular. La demo de fuga muestra un tono a −50 dB que la rectangular tapa y la Hann deja ver.

**¿Que es COLA y que estaba mal antes?**
Que la suma de las ventanas desplazadas sea constante. La cumple la Hann periodica con salto N/R (R entero ≥ 2), no la simetrica. Antes se usaba la simetrica y se afirmaba que cumplia; ahora se usa la periodica con salto N/3 y la suma vale exactamente 1.5 (hay test).

**¿Como funciona la FFT que implementaron?**
Radix-2 por decimacion en el tiempo: reordenar por inversion de bits y hacer log2(N) etapas de mariposas $E \pm W^k O$. Para N = 512 son 9 etapas, unas 57 veces menos operaciones que la DFT directa. Coincide con `np.fft` con error menor a $10^{-9}$ y da las mismas caracteristicas.

**¿Que representan los MFCC?**
La envolvente espectral en escala perceptual: energias en 40 bandas mel, logaritmo (separa fuente y filtro y comprime como el oido) y DCT (decorrela y concentra la forma en pocos coeficientes). Los primeros 13 describen formantes y timbre sin la estructura fina de los armonicos.

**¿Que aporta la planitud espectral?**
Es media geometrica sobre media aritmetica del espectro: 1 para ruido blanco, cerca de 0 para un espectro de pocas lineas. Separa directamente ruido de senales armonicas (voz sonora, musica).

**¿Por que media y desvio sobre 3 s?**
Un frame no alcanza para decidir. La voz se reconoce por la variacion: alterna sonoras y fricativas, tiene pausas; eso aparece en el desvio de ZCR, de flujo y de deltas, y en la fraccion de frames de baja energia.

**¿Por que ventana de 3 s cada 0.5 s?**
3 s dan contexto suficiente para estadisticas estables; el salto de 0.5 s da resolucion temporal de 0.5 s sin achicar la ventana. El precio es que un borde se ubica con una incertidumbre del orden de la mitad de la ventana.

**¿Que hace el Viterbi y como se elige la penalidad?**
Busca la secuencia de clases que maximiza la suma de log-probabilidades menos λ por cada cambio; es un HMM con probabilidad de cambio ε = (K−1)/(e^λ + K−1). Con λ = 2 un pico de una o dos celdas no alcanza (tiene que pagar dos cambios), tres celdas seguidas si. En 30 secuencias del pliegue de prueba, λ = 2 dio el mejor acierto por celda (0.876 contra 0.870 sin suavizado) y bajo los segmentos de 282 a 240 con 244 reales. Penalidades mas altas absorbian tramos reales. Se eligio mirando esas secuencias, asi que es algo optimista.

**¿Por que Viterbi y no solo mediana?**
La mediana decide celda por celda. Viterbi optimiza toda la secuencia con un costo explicito por cambio: es el estimador MAP de un modelo con un supuesto claro (los segmentos duran). Ademas tiene una version causal para el modo en vivo con la misma matriz de transicion.

**¿Que pasa con los ultimos segundos del audio?**
Se agrega una ventana final alineada con el fin del audio, asi la cola se clasifica. Si el audio entero dura menos de 3 s, se repite hasta completar la ventana (no se rellena con ceros porque eso parece silencio).

**¿Como evitaron el data leakage?**
Particion por archivo (y por artista en musica, que MUSAN informa) en 5 pliegues; el audio sintetico de cada pliegue sale solo de ese pliegue; el escalado se ajusta solo con entrenamiento; el pliegue de prueba no se toca hasta el final.

**¿Por que F1 macro?**
Porque pesa igual las cuatro clases. Con accuracy un modelo puede ignorar la clase minoritaria (ruido) y seguir dando un numero alto.

**¿Como separan silencio de ruido?**
Principalmente por nivel: silencio es senal sin eventos por debajo de −36 dBFS (medido despues del filtro) y el ruido de entrenamiento empieza en −34 dBFS. Tambien ayuda la forma: el silencio de un celular con AGC es casi todo grave, muy poco plano y con flujo alto. Por eso no se normaliza la amplitud. El umbral se corrigio despues de medir que un celular con AGC deja el silencio de una pieza en unos −41 dBFS.

**¿Como simularon un celular?**
Reverb sintetica, pasa-banda (300–3400 Hz o 100–7000 Hz), ganancia y saturacion, y codecs reales con ffmpeg (AMR-NB, MP3, AAC, Opus). Se entrena con eso y se mide aparte cuanto cae la accuracy con cada codec.

**¿Cuanto mejoro respecto de la version anterior?**
Sobre los mismos recortes: canal de celular completo de 0.742 a 0.939; AMR-NB de 0.663 a 0.871; SNR de 10 dB de 0.732 a 0.928; 8 kHz de 0.654 a 0.905. El recall de ruido en la prueba paso de 0.659 a 0.871 (con otra particion, mas dificil). En grabaciones reales de celular, de 0.379 a 0.678 por celda. Ademas la resolucion temporal paso de 3 s a 0.5 s y la cola se clasifica.

**¿Por que Gradient Boosting y no SVM, si la diferencia es minima?**
Porque el criterio (F1 macro en CV) se fijo antes de entrenar y GB quedo primero: 0.944 contra 0.938. La diferencia es del orden de un desvio, asi que es honesto decir que son practicamente equivalentes. El modelo guardado pesa 1.9 MB.

**¿Por que falla el silencio grabado con el celular?**
Porque el control automatico de ganancia sube el ruido de la pieza a −41 dBFS, con una forma que varia en el tiempo: queda indistinguible de un ruido ambiente bajo. Silencio y ruido se separan por nivel, y el AGC le quita significado al nivel absoluto. Se intento corregir con datos sinteticos y no alcanzo sin ajustar el modelo a ese archivo. La solucion de fondo seria grabar sin AGC o medir el silencio en forma relativa dentro de la misma grabacion.

**¿Cual es la limitacion principal?**
La resolucion de los bordes (~1 s, por la ventana de 3 s) y la definicion de silencio por nivel: depende de la ganancia del dispositivo. Con AGC el silencio queda en unos −41 dBFS (por eso el umbral de −36); un dispositivo todavia mas agresivo podria superarlo, y un ruido continuo pero muy bajo (un ventilador lejano) se va a etiquetar como silencio, que es coherente con la definicion pero discutible. Tambien la musica con voz cantada contra voz hablada con fondo musical, que es ambigua incluso para una persona.
