---
title: "Fundamentos Teoricos y Procedimiento del Sistema de Segmentacion y Clasificacion de Audio"
date: "2026-11-06"
---

# Sistema de Segmentacion y Clasificacion Automatica de Audio

## 1. Arquitectura General y Jerarquia Temporal

El procesamiento de señales de audio para clasificacion acustica enfrenta un compromiso fundamental entre estacionariedad estadistica y discriminabilidad semantica:

1. **No estacionariedad global:** Una señal de audio de larga duracion cambia sus propiedades espectrales y temporales continuamente. Aplicar una Transformada de Fourier sobre varios segundos mezcla eventos acusticos distintos y destruye la estructura temporal.
2. **Cuasi-estacionariedad local (Corto plazo):** En intervalos de entre $20\text{ ms}$ y $40\text{ ms}$, el tracto vocal humano y los instrumentos musicales mantienen una configuracion fisica aproximadamente constante.
3. **Contexto estadistico (Medio plazo):** Un unico frame de $32\text{ ms}$ no permite distinguir entre musica, habla o ruido, ya que una consonante fricativa aislada se asemeja al ruido blanco y una pausa breve entre silabas parece silencio. Es necesario observar la evolucion estadistica (media y dispersion) de las caracteristicas de corto plazo a lo largo de una ventana de decision de varios segundos.

Por ello, el sistema implementa una jerarquia de dos escalas temporales operando a una frecuencia de muestreo $f_s = 16000\text{ Hz}$:

| Nivel Temporal | Simbolo | Duracion | Muestras ($f_s = 16\text{ kHz}$) | Funcion en el Pipeline |
| :--- | :---: | :---: | :---: | :--- |
| Segmento de decision (Medio plazo) | $T_{\text{seg}}$ | $3.0\text{ s}$ | $48000$ | Unidad de clasificacion y agregacion estadistica ($\mu, \sigma$) |
| Ventana de analisis (Corto plazo) | $N_{\text{win}}$ | $32\text{ ms}$ | $512$ | Unidad de analisis cuasi-estacionario para calculo temporal y RFFT |
| Salto entre ventanas (Hop length) | $H$ | $16\text{ ms}$ | $256$ | Desplazamiento con $50\%$ de solapamiento para evitar perdida en bordes |

El numero total de frames $M$ contenidos en cada segmento de $L = 48000$ muestras viene dado por:

$$M = 1 + \left\lfloor \frac{L - N_{\text{win}}}{H} \right\rfloor = 1 + \left\lfloor \frac{48000 - 512}{256} \right\rfloor = 186 \text{ frames}$$

---

## 2. Configuracion del Sistema (`src/config.py`)

El modulo `src/config.py` centraliza los parametros fisicos, matematicos y estructurales del sistema para garantizar consistencia entre entrenamiento e inferencia:

- **Frecuencia de muestreo y cuantizacion:** `SAMPLE_RATE = 16000` establece una frecuencia de Nyquist de $f_{\text{Nyquist}} = \frac{f_s}{2} = 8000\text{ Hz}$. `PCM_MAX_VALUE = 32768.0` corresponde a la magnitud maxima representable en entero con signo de 16 bits ($2^{15}$).
- **Division en subbandas de octava:** El rango espectral util $[0, 8000]\text{ Hz}$ se particiona en 5 subbandas siguiendo una progresion geometrica por octavas:
  - Banda 1: $[0, 500]\text{ Hz}$ (frecuencia fundamental de la voz, graves profundos y energia de baja frecuencia).
  - Banda 2: $[500, 1000]\text{ Hz}$ (primer formante vocal y cuerpo armonico).
  - Banda 3: $[1000, 2000]\text{ Hz}$ (segundo formante vocal, determinante en inteligibilidad del habla).
  - Banda 4: $[2000, 4000]\text{ Hz}$ (tercer formante, consonantes y brillo instrumental).
  - Banda 5: $[4000, 8000]\text{ Hz}$ (consonantes fricativas/sibilantes y componentes de alta frecuencia de ruido o percusiones).
- **Estabilidad numerica:** `EPSILON = 1e-12` evita indeterminaciones del tipo $\frac{0}{0}$ cuando se procesan tramas con energia nula o cercana al cero absoluto.

---

## 3. Preprocesamiento de la Señal (`src/preprocessing.py`)

### 3.1. Escalado de Cuantizacion PCM de 16 bits

Los archivos `.wav` del dataset MUSAN almacenan las muestras como enteros con signo de 16 bits en complemento a dos ($x_{\text{int16}}[n] \in [-32768, 32767]$). La conversion al dominio continuo en punto flotante preserva la relacion absoluta de amplitud:

$$x[n] = \frac{x_{\text{int16}}[n]}{32768.0}, \quad x[n] \in [-1.0, 1.0)$$

### 3.2. Eliminacion del Nivel DC (DC Offset) sin Normalizacion de Pico

Un convertidor analogico-digital puede introducir una componente continua (frecuencia $0\text{ Hz}$) que desplaza la señal respecto al eje cero. Si no se corrige, este sesgo distorsiona la Tasa de Cruces por Cero (ZCR) y añade energia espuria en el bin $k = 0$ de la FFT.

Para cada segmento de longitud $L$, se resta su valor medio:

$$\tilde{x}[n] = x[n] - \frac{1}{L} \sum_{i=0}^{L-1} x[i]$$

> [!important]
> No se aplica normalizacion de amplitud pico ($\frac{\tilde{x}[n]}{\max |\tilde{x}[n]|}$). Normalizar a escala completa amplificaria el ruido termico o de cuantizacion de los tramos de silencio hasta niveles equivalentes a los de la voz o la musica, destruyendo el poder discriminativo de la energia de corto plazo.

### 3.3. Entramado (Framing) y Ventana de Hann

Extraer un bloque finito de $N_{\text{win}}$ muestras sin ponderar equivale a multiplicar la señal infinita por una ventana rectangular $w_R[n]$. En el dominio de la frecuencia, la multiplicacion temporal corresponde a la convolucion circular con la transformada de la ventana rectangular (funcion Dirichlet o $\text{sinc}$), cuyos lobulos laterales caen lentamente a razon de $-6\text{ dB/octava}$, produciendo fuga espectral (*spectral leakage*).

Para mitigar la fuga espectral y suavizar las discontinuidades en los extremos de cada frame, se multiplica cada trama por una ventana de Hann definida analiticamente como:

$$w[n] = 0.5 \left( 1 - \cos\left( \frac{2\pi n}{N_{\text{win}} - 1} \right) \right), \quad 0 \le n \le N_{\text{win}} - 1$$

La ventana de Hann presenta lobulos laterales que decaen a $-18\text{ dB/octava}$. Dado que la ventana atenua la señal hacia los bordes hasta anularse en $n = 0$ y $n = N_{\text{win}} - 1$, se emplea un solapamiento del $50\%$ ($H = \frac{N_{\text{win}}}{2} = 256$) cumpliendo la condicion de suma constante (COLA, *Constant Overlap-Add*), de modo que ninguna region temporal de la señal pierda representacion energetica.

El $m$-esimo frame enventanado (para $m = 0, 1, \dots, M-1$) se expresa como:

$$x_m[n] = \tilde{x}[mH + n] \cdot w[n], \quad 0 \le n \le N_{\text{win}} - 1$$

---

## 4. Extraccion de Caracteristicas (`src/features.py`)

Para cada frame $m$ de un segmento se calculan descriptores temporales y espectrales.

### 4.1. Dominio Temporal

1. **Energia RMS (Root Mean Square):** Mide la potencia acustica instantanea del frame. Permite separar eficazmente el silencio del sonido activo y capturar la modulacion de amplitud silabica propia del habla frente a la energia mas sostenida de la musica:
   $$\text{RMS}_m = \sqrt{\frac{1}{N_{\text{win}}} \sum_{n=0}^{N_{\text{win}}-1} x_m^2[n]}$$

2. **Tasa de Cruces por Cero (ZCR, Zero-Crossing Rate):** Mide la fraccion de muestras adyacentes en las que la señal cambia de signo algebraico (calculada sobre los frames sin ventana o con ventana positiva):
   $$\text{ZCR}_m = \frac{1}{2(N_{\text{win}} - 1)} \sum_{n=1}^{N_{\text{win}}-1} \left| \text{sgn}(\tilde{x}_m[n]) - \text{sgn}(\tilde{x}_m[n-1]) \right|$$
   La voz presenta alternancia marcada entre fonemas sonoros (bajo ZCR) y consonantes fricativas (alto ZCR), lo que produce una alta desviacion estandar de ZCR a lo largo del segmento de 3 segundos.

### 4.2. Dominio Espectral (RFFT)

Dado que $x_m[n] \in \mathbb{R}$, su Transformada Discreta de Fourier presenta simetria hermitica ($X_m[N_{\text{win}} - k] = X_m^*[k]$). Basta calcular la mitad no redundante mediante `np.fft.rfft`, obteniendo $K = \frac{N_{\text{win}}}{2} + 1 = 257$ bins de frecuencia asociados a las frecuencias fisicas:

$$f[k] = k \cdot \frac{f_s}{N_{\text{win}}} = k \cdot 31.25\text{ Hz}, \quad k = 0, 1, \dots, 256$$

Sea $S_m[k] = |X_m[k]|$ el espectro de magnitud y $E_m[k] = |X_m[k]|^2$ el espectro de densidad de energia del frame $m$.

1. **Centroide Espectral ($C_m$):** Representa el centro de gravedad del espectro de magnitud y se correlaciona perceptualemente con el brillo del sonido:
   $$C_m = \frac{\sum_{k=0}^{K-1} f[k] \cdot S_m[k]}{\sum_{k=0}^{K-1} S_m[k] + \epsilon}$$

2. **Ancho de Banda Espectral ($B_m$):** Corresponde al momento central de segundo orden (desviacion estandar espectral) alrededor del centroide $C_m$. Cuantifica si la energia espectral esta concentrada en pocos armonicos o dispersa en todo el espectro (como en el ruido de banda ancha):
   $$B_m = \sqrt{\frac{\sum_{k=0}^{K-1} (f[k] - C_m)^2 \cdot S_m[k]}{\sum_{k=0}^{K-1} S_m[k] + \epsilon}}$$

3. **Energia Relativa en Subbandas ($R_{m,b}$):** Para cada una de las 5 bandas de octava $b = [f_{\text{low}}^{(b)}, f_{\text{high}}^{(b)}]$, sea $\mathcal{K}_b = \{ k \mid f_{\text{low}}^{(b)} \le f[k] < f_{\text{high}}^{(b)} \}$ el conjunto de indices espectrales pertenecientes a la banda. La proporcion de energia contenida en la subbanda $b$ respecto a la energia espectral total del frame es:
   $$R_{m,b} = \frac{\sum_{k \in \mathcal{K}_b} E_m[k]}{\sum_{k=0}^{K-1} E_m[k] + \epsilon}$$

### 4.3. Agregacion Estadistica a Nivel de Segmento

Por cada frame $m \in \{0, \dots, 185\}$ se obtiene un vector instantaneo $\mathbf{v}_m \in \mathbb{R}^{9}$ compuesto por:
- 2 descriptores temporales: $\text{RMS}_m$, $\text{ZCR}_m$
- 2 descriptores espectrales globales: $C_m$, $B_m$
- 5 razones de energia por subbanda: $R_{m,1}, R_{m,2}, R_{m,3}, R_{m,4}, R_{m,5}$

Para capturar tanto el comportamiento estacionario como la dinamica temporal a lo largo de los $3.0\text{ s}$, se calculan la media ($\boldsymbol{\mu}$) y la desviacion estandar ($\boldsymbol{\sigma}$) a traves de los $M = 186$ frames, concatenandolas en un vector de caracteristicas final $\mathbf{x} \in \mathbb{R}^{18}$:

$$\mathbf{x} = \left[ \mu_1, \mu_2, \dots, \mu_9, \sigma_1, \sigma_2, \dots, \sigma_9 \right]^T$$

---

## 5. Construccion del Dataset y Prevencion de Data Leakage (`src/dataset.py`)

### 5.1. Particion a Nivel de Archivo

Si un archivo de audio de 3 minutos se corta primero en 60 segmentos de 3 segundos y luego se reparten aleatoriamente entre entrenamiento y prueba, segmentos contiguos del mismo locutor, cancion o fuente de ruido caeran en ambos conjuntos. El clasificador memorizaria el timbre del canal o del locutor en lugar de aprender la categoria acustica (*data leakage*).

Por ello, la particion entrenamiento/prueba ($80\% / 20\%$) se realiza estrictamente sobre la lista de archivos `.wav` originales antes de extraer cualquier segmento.

### 5.2. Sintesis de la Clase Silencio

En grabaciones reales, el silencio absoluto ($x[n] = 0$) rara vez existe; en su lugar aparece ruido de piso del preamplificador o sonido ambiente de muy baja energia. Para modelar esta clase de forma realista sin fugas entre particiones:
1. Se seleccionan tramos de baja energia (debajo del percentil 10 de energia RMS) provenientes de archivos de voz del mismo subconjunto (train o test respectivamente).
2. Se toman segmentos de ruido estacionario del mismo subconjunto y se escalan por un factor aleatorio uniforme $\alpha \sim \mathcal{U}(0.001, 0.01)$, equivalente a una atenuacion de entre $-40\text{ dB}$ y $-60\text{ dB}$.

---

## 6. Normalizacion, Entrenamiento e Inferencia (`src/train.py` y `src/segmenter.py`)

### 6.1. Estandarizacion de Caracteristicas

El vector $\mathbf{x} \in \mathbb{R}^{18}$ combina magnitudes con escalas fisicas dispares: la energia RMS y el ZCR toman valores en $[0, 1]$, mientras que el centroide y el ancho de banda espectral alcanzan miles de Hercios ($[0, 8000]\text{ Hz}$). Sin normalizacion, las distancias euclideas en k-NN y los gradientes iniciales en el MLP estarian dominados exclusivamente por las variables en Hercios.

Se aplica `StandardScaler` ajustando la media $\boldsymbol{\mu}_{\text{train}}$ y desviacion estandar $\boldsymbol{\sigma}_{\text{train}}$ unicamente sobre el conjunto de entrenamiento:

$$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$

### 6.2. Modelos Clasificadores

- **K-Nearest Neighbors (`KNeighborsClassifier`):** Clasificador no parametrico basado en instancias que asigna la clase mayoritaria entre los $k$ vecinos mas cercanos en el espacio caracteristico $\mathbb{R}^{18}$ estandarizado.
- **Perceptron Multicapa (`MLPClassifier`):** Red neuronal prealimentada con capas ocultas densas y activacion no lineal ReLU, capaz de aprender fronteras de decision no lineales complejas entre clases acusticas superpuestas.

### 6.3. Segmentacion de Audio Continuo

Dado un archivo `.wav` de duracion arbitraria $T_{\text{total}}$, el modulo `src/segmenter.py`:
1. Carga la señal a $16\text{ kHz}$ y la escala a $[-1.0, 1.0]$.
2. La divide en $N_{\text{seg}} = \lfloor \frac{T_{\text{total}}}{3.0} \rfloor$ bloques consecutivos de $3.0\text{ s}$.
3. Remueve el DC offset por bloque, aplica entramado y extrae la matriz de caracteristicas $\mathbf{X} \in \mathbb{R}^{N_{\text{seg}} \times 18}$.
4. Transforma $\mathbf{X}$ mediante el `StandardScaler` persistido y predice la secuencia de etiquetas $\hat{y}_i \in \{\text{musica}, \text{voz}, \text{ruido}, \text{silencio}\}$, devolviendo los intervalos temporales $[t_{\text{inicio}}, t_{\text{fin}}]$ con su respectiva categoria.
