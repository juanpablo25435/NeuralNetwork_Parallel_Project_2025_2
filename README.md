# Implementación Paralela de Red Neuronal (MLP) - Proyecto Final

Este repositorio contiene 5 implementaciones de una red neuronal multicapa para clasificar el dataset MNIST, diseñadas para comparar rendimiento entre ejecución secuencial, paralela en CPU y paralela en GPU.

## Estructura del Proyecto
* `src/1_sequential_python`: Implementación de referencia (NumPy).
* `src/2_sequential_c`: Implementación base en C puro.
* `src/3_parallel_python`: Implementación usando `multiprocessing`.
* `src/4_parallel_c_openmp`: Implementación optimizada con OpenMP.
* `src/5_gpu_cuda`: Implementación de alto rendimiento con CUDA C++.
* `data/`: Scripts de descarga y dataset MNIST.

## Instrucciones de Compilación y Ejecución

### Prerrequisitos
1. Python 3.8+ con `numpy` y `matplotlib`.
2. Compilador GCC (Linux/Windows WSL/MinGW).
3. Entorno NVIDIA CUDA (Para la versión GPU, se recomienda Google Colab).

### Paso 0: Datos
```bash
python data/download_mnist.py
gunzip -k data/*.gz  # Necesario para las versiones en C

### Paso 1: Python Secuencial
```bash
cd src/1_sequential_python
py main.py

### Paso 2: C Secuencial
```bash
cd src/2_sequential_c
make
./mlp_seq

### Paso 3: Python Multiprocessing
```bash
cd src/3_parallel_python
py main_mp.py

### Paso 4: C OpenMP
```bash
cd src/4_parallel_c_openmp
make
# Para variar hilos: export OMP_NUM_THREADS=4
./mlp_omp

### Paso 5: GPU CUDA (Ejecutar en Google Colab)
# NOTA: Se deben adjuntar los archivos de la carpeta src/5_gpu_cuda  y common en Google Colab (GPU T4) y correr el codigo de ParalelizacionGPU.ipynb

### Paso 6: Generar Gráficas
```bash
python generar_graficas.py
