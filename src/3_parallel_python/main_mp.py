import sys
import os
import time
import numpy as np
import gzip
from multiprocessing import Pool, cpu_count
from parallel_mlp import ParallelMLP

# Rutas
DATA_PATH = '../../data'

def load_images(filename):
    path = os.path.join(DATA_PATH, filename)
    with gzip.open(path, 'rb') as f:
        f.read(16)
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        return data.reshape(-1, 784) / 255.0

def load_labels(filename):
    path = os.path.join(DATA_PATH, filename)
    with gzip.open(path, 'rb') as f:
        f.read(8)
        return np.frombuffer(buffer=f.read(), dtype=np.uint8)

def one_hot_encode(y, classes=10):
    return np.eye(classes)[y].T

def main():
    print("--- Escenario 2a: Python Multiprocessing ---")
    num_cpus = cpu_count()
    print(f">>> Usando Pool de {num_cpus} procesos <<<")
    
    # Cargar datos
    try:
        X_train = load_images('train-images-idx3-ubyte.gz').T
        y_train = load_labels('train-labels-idx1-ubyte.gz')
        Y_train_enc = one_hot_encode(y_train)
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return

    # Configuración
    mlp = ParallelMLP(784, 512, 10, 0.1)
    EPOCHS = 5  # Reducimos a 5 epochs porque MP en Windows tiene mucho overhead de inicio
    BATCH_SIZE = 64
    
    print(f"Iniciando entrenamiento ({EPOCHS} epochs)...")
    start_time = time.time()
    
    # Crear el Pool de procesos UNA VEZ
    # En Windows esto puede tardar unos segundos en arrancar
    with Pool(processes=num_cpus) as pool:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            mlp.train_epoch_parallel(pool, X_train, Y_train_enc, BATCH_SIZE)
            print(f"Epoch {epoch+1} terminada en {time.time() - epoch_start:.2f}s")
            
    total_time = time.time() - start_time
    print(f"\n>>> Tiempo Total (Python MP): {total_time:.2f} segundos <<<")
    print("(Nota: Compara esto con el Baseline Python Secuencial)")

if __name__ == '__main__':
    main()