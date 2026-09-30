#  Instituto Tecnológico de Costa Rica
#  Escuela de Ingeniería Electrónica
#  Curso Introduccion a la Computacion Heterogenea 
#  Profesor Dr. Luis Gerardo León vega
#  Estudiante rodrigo Venegas Mora
#  Practica semana /


## Ejercicio A: Suma de vectores (`vector-add/`)

### Código Completo

```cuda
__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### Compilación y ejecución

```bash
cd vector-add
make clean
make
make run
```

### Salida obtenida

```
nvcc -O2 -std=c++17 -lineinfo -o vector_add main.cu
./vector_add 1048576
vector-add n=1048576: OK
```

### Respuestas

**1. ¿Cuántos bloques se lanzan cuando N=1048576 y cada bloque tiene 256 hilos?**

Se lanzan **4096 bloques**. El cálculo es:

```
blocks = ceil(N / threads_per_block) = ceil(1048576 / 256) = 4096
```

---

**2. ¿Qué ocurre si N no es múltiplo del tamaño del bloque?**

Cuando N no es múltiplo de 256, el número de bloques se calcula con redondeo
hacia arriba: `blocks = ceil(N / 256)`. En caso de que el último bloque tenga
hilos sobrantes cuyos índices globales cumplen `i >= N`.

Si se elimina la condición `if (i < n)`:

- Los hilos sobrantes escribirían en c[i] fuera del rango reservado,
  provocando corrupción de memoria o un segmentation fault.
- También leerían posiciones inválidas de a[i] y b[i], obteniendo valores
  basura.

Por esta razón i < n es obligatoria siempre que el tamaño del
problema no sea múltiplo exacto del tamaño del bloque.

---

**3. ¿Qué transferencias de memoria ocurren entre CPU y GPU?**

| # | Dirección | Qué se transfiere | Llamada CUDA |
|---|-----------|-------------------|--------------|
| 1 | Host → Device | Vector `a` | `cudaMemcpy(..., cudaMemcpyHostToDevice)` |
| 2 | Host → Device | Vector `b` | `cudaMemcpy(..., cudaMemcpyHostToDevice)` |
| 3 | Device → Host | Vector `c` (resultado) | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` |


En total: 2 transferencias Host → Device y 1 transferencia Device → Host.

## Ejercicio B: Producto punto (`dot-product/`)

### Código del kernel completado

```cuda
__global__ void dot_partial_kernel(const float *a, const float *b,
                                   float *partials, int n) {
    extern __shared__ float cache[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    if (i < n) {
        value = a[i] * b[i];
    }

    cache[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = cache[0];
    }
}
```

### Compilación y ejecución

```bash
cd dot-product
make clean
make
make run N=1048576
make run N=4194304
```

### Salida obtenida

```
dot-product n=1048576: gpu=-21.250000 cpu=-21.250000 error=0.000000 OK
dot-product n=4194304: gpu=-0.500000 cpu=-0.500000 error=0.000000 OK
```

### Respuestas

**1. ¿Por qué este ejercicio no puede resolverse solamente escribiendo un valor
independiente por hilo?**

Porque el resultado del producto punto es un único escalar que combina los
productos de todos los pares a[i] * b[i]. A diferencia de la suma de vectores,
donde cada hilo escribe su resultado en una posición independiente de c[],todos
 los hilos deben combinar sus valores parciales en un solo número.
Esa combinación es una operación es una reducción y requiere
que los hilos del bloque cooperen entre sí usando memoria compartida y
sincronización. Sin reducción, cada hilo solo podría calcular a[i] * b[i] pero
no se sumarían todos esos productos.

---

**2. ¿Cuántos valores parciales se copian de GPU a CPU?**

Se copia un valor parcial por bloque. El número de bloques es:

```
blocks = ceil(N / 256)
```

- Para N = 1048576 : blocks = 1048576 / 256 = 4096 valores parciales.
- Para N = 4194304 : blocks = 4194304 / 256 = 16384 valores parciales.

Cada bloque escribe partials[blockIdx.x] = cache[0], por lo que el arreglo
partials tiene exactamente blocks elementos. La CPU luego los suma
secuencialmente para obtener el resultado final.

---

**3. ¿Qué pasaría si se elimina alguna sincronización dentro de la reducción?**

Se produciría  condiciones de carrera (*race conditions*). El bucle de
reducción funciona en etapas:

```
stride = 128 → cache[0..127] += cache[128..255]
stride =  64 → cache[0.. 63] += cache[ 64..127]
stride =  32 → cache[0.. 31] += cache[ 32.. 63]
...
```

Cada etapa depende de que la etapa anterior haya terminado en todos los
hilos. Sin __syncthreads() entre etapas. Un hilo podría leer cache[tid + stride] 
antes de que otro hilo haya actualizado ese valor en la etapa anterior.
El resultado sería incorrecto y no determinista, porque depende del orden
 en que el planificador ejecute los warpsy en el peor caso, el resultado 
final diferiría entre ejecuciones.

syncthreads() garantiza que todos los hilos del bloque lleguen a la barrera
antes de continuar, asegurando que cada etapa vea los datos ya actualizados por
la etapa anterior.

## Ejercicio C: Softmax (`softmax/`)

### Código del kernel completado

```cuda
__global__ void softmax_rows_kernel(const float *input, float *output, int rows,
                                    int cols) {
    extern __shared__ float cache[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    // ---------- 1) Máximo de la fila ----------
    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row * cols + col]);
    }

    cache[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] = fmaxf(cache[tid], cache[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = cache[0];

    // ---------- 2) Exponenciales desplazadas ----------
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        float e = expf(input[idx] - row_max);
        output[idx] = e;
        local_sum += e;
    }

    cache[tid] = local_sum;
    __syncthreads();

    // ---------- 3) Suma de exponenciales ----------
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = cache[0];

    // ---------- 4) Normalizar ----------
    for (int col = tid; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        output[idx] /= row_sum;
    }
}
```

### Compilación y ejecución

```bash
cd softmax
make clean
make
make run ROWS=256 COLS=2048
```

### Salida obtenida

```
nvcc -O2 -std=c++17 -lineinfo -o softmax main.cu
./softmax 256 2048
softmax rows=256 cols=2048: OK
```

### Respuestas

**1. ¿Por qué se calcula primero el máximo de cada fila?**

Por **estabilidad numérica**. La función `exp(x)` crece muy rápido: para
`x > ~88` en `float` ya devuelve `inf`. Si la fila tuviera valores grandes (por
ejemplo `100`, `200`, `300`), al calcular `exp(100)` se desbordaría y el
resultado sería `inf` o `NaN` tras dividir `inf / inf`.

Restando el máximo de la fila antes de exponenciar:

```
y = exp(x - max)
```

nos aseguramos de que el mayor exponente sea `exp(0) = 1`, y todos los demás
son `≤ 1`. Ningún término se desborda. Matemáticamente el resultado es el mismo,
porque:

```
softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)
```

El factor `exp(-m)` aparece tanto en el numerador como en el denominador y se
cancela.

---

**2. ¿Qué partes del algoritmo requieren cooperación entre hilos del mismo bloque?**

Tres partes:

- **Reducción del máximo de la fila**: todos los hilos aportan su máximo local y
  se combinan en `cache[]` con `fmaxf`.
- **Reducción de la suma de exponenciales**: todos los hilos aportan su suma
  local y se combinan con `+=`.
- **Normalización**: cada hilo divide sus elementos entre `row_sum`, que fue
  calculado colectivamente.

Las tres requieren `__syncthreads()` entre etapas para evitar condiciones de
carrera.

---

**3. ¿Qué limitación tiene usar un solo bloque por fila cuando `cols` crece mucho?**

Varias limitaciones:

- **Tamaño máximo del bloque**: una GPU típica admite hasta **1024 hilos por
  bloque**. Si `cols` es mucho mayor (por ejemplo 1 000 000), cada hilo debe
  procesar muchas columnas en un bucle, y la reducción solo usa 1024 hilos →
  subutilización.
- **Memoria compartida limitada**: aunque el `cache[]` solo tiene
  `blockDim.x` floats, el patrón un-bloque-por-fila desperdicia recursos cuando
  `cols` es muy grande, porque la mayor parte del trabajo ocurre en bucles
  seriales por hilo.
- **Ocupancia baja**: si `rows` es pequeño y `cols` enorme, hay pocos bloques
  lanzados, y la GPU queda mayoritariamente ociosa.
- **Escalabilidad limitada**: el algoritmo no aprovecha el paralelismo masivo
  disponible cuando `cols` supera ampliamente el tamaño del bloque.

En esos casos conviene **dividir la fila en varios bloques** y combinar los
resultados con una segunda pasada o una operación atómica, o usar `grid-stride
loops` con más hilos cooperando.