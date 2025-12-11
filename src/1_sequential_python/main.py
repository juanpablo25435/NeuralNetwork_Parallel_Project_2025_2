import sys
import os
import gzip
import time
import numpy as np
from mlp import MLP

# Ajustar rutas para encontrar la carpeta data
DATA_PATH = '../../data'

def load_images(filename):
    path = os.path.join(DATA_PATH, filename)
    with gzip.open(path, 'rb') as f:
        f.read(16) # Skip headers
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        # Normalizar [0, 1] y aplanar a (N, 784)
        return data.reshape(-1, 784) / 255.0

def load_labels(filename):
    path = os.path.join(DATA_PATH, filename)
    with gzip.open(path, 'rb') as f:
        f.read(8) # Skip headers
        return np.frombuffer(buffer=f.read(), dtype=np.uint8)

def one_hot_encode(y, classes=10):
    return np.eye(classes)[y].T # Transponer para obtener (10, N)

def get_accuracy(predictions, Y):
    # predictions shape: (10, m)
    # Y shape: (10, m)
    pred_labels = np.argmax(predictions, axis=0)
    true_labels = np.argmax(Y, axis=0)
    return np.sum(pred_labels == true_labels) / Y.shape[1]

def main():
    print("--- Escenario 1: Baseline Secuencial (Python) ---")
    
    # 1. Cargar Datos
    print("Cargando dataset MNIST...")
    try:
        X_train = load_images('train-images-idx3-ubyte.gz')
        y_train = load_labels('train-labels-idx1-ubyte.gz')
        X_test = load_images('t10k-images-idx3-ubyte.gz')
        y_test = load_labels('t10k-labels-idx1-ubyte.gz')
    except FileNotFoundError:
        print("Error: No se encuentran los datos. Ejecuta primero data/download_mnist.py")
        return

    # Preparar datos para la red (Features x Samples)
    # X debe ser (784, 60000)
    X_train = X_train.T
    X_test = X_test.T
    
    # Y debe ser One-Hot (10, 60000)
    Y_train_enc = one_hot_encode(y_train)
    Y_test_enc = one_hot_encode(y_test)
    
    print(f"Datos Cargados. X_train: {X_train.shape}, Y_train: {Y_train_enc.shape}")
    
    # 2. Configuración
    INPUT_SIZE = 784
    HIDDEN_SIZE = 512 # Recomendado entre 256 y 1024
    OUTPUT_SIZE = 10
    LEARNING_RATE = 0.1
    EPOCHS = 10
    BATCH_SIZE = 64
    
    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE)
    
    # 3. Entrenamiento
    print(f"Entrenando por {EPOCHS} épocas...")
    start_time = time.time()
    
    m = X_train.shape[1]
    
    for epoch in range(EPOCHS):
        # Shuffle
        permutation = np.random.permutation(m)
        X_shuffled = X_train[:, permutation]
        Y_shuffled = Y_train_enc[:, permutation]
        
        for i in range(0, m, BATCH_SIZE):
            end = i + BATCH_SIZE
            X_batch = X_shuffled[:, i:end]
            Y_batch = Y_shuffled[:, i:end]
            
            # Forward & Backward
            mlp.forward(X_batch)
            mlp.backward(X_batch, Y_batch)
        
        # Calcular Loss y Accuracy al final de la época (consume tiempo)
        # Hacemos un forward sobre una muestra pequeña para monitorear rápido
        sample_out = mlp.forward(X_train[:, :1000])
        acc = get_accuracy(sample_out, Y_train_enc[:, :1000])
        print(f"Epoch {epoch+1}/{EPOCHS} - Acc (aprox): {acc:.4f}")

    total_time = time.time() - start_time
    print(f"Entrenamiento finalizado en {total_time:.2f} segundos.")
    
    # 4. Evaluación Final
    print("Evaluando en set de prueba...")
    test_out = mlp.forward(X_test)
    test_acc = get_accuracy(test_out, Y_test_enc)
    print(f"Accuracy Final en Test: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()