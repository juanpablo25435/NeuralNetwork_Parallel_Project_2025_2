import numpy as np
import time
from multiprocessing import Pool, cpu_count

# Funciones auxiliares fuera de la clase para que multiprocessing no llore
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def compute_gradients(args):
    """
    Esta función es la que ejecuta cada núcleo.
    Recibe: (Pesos, X_chunk, Y_chunk)
    Devuelve: (dW1, db1, dW2, db2)
    """
    W1, b1, W2, b2, X, Y = args
    m = X.shape[1]
    
    # --- Forward ---
    Z1 = np.dot(W1.T, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2.T, A1) + b2
    A2 = softmax(Z2)
    
    # --- Backward ---
    dZ2 = A2 - Y
    
    dW2 = np.dot(A1, dZ2.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = np.dot(W2, dZ2)
    dZ1 = dA1 * (Z1 > 0) # Derivada ReLU
    
    dW1 = np.dot(X, dZ1.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    # Devolvemos los gradientes y el tamaño del chunk para promediar después
    return (dW1, db1, dW2, db2, m)

class ParallelMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Inicialización
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def train_epoch_parallel(self, pool, X_train, Y_train, batch_size):
        m = X_train.shape[1]
        
        # Shuffle
        perm = np.random.permutation(m)
        X_shuffled = X_train[:, perm]
        Y_shuffled = Y_train[:, perm]
        
        # Iterar sobre minibatches
        for i in range(0, m, batch_size):
            end = min(i + batch_size, m)
            X_batch = X_shuffled[:, i:end]
            Y_batch = Y_shuffled[:, i:end]
            
            # --- ESTRATEGIA PARALELA ---
            # Dividimos el batch en partes iguales para cada CPU
            num_processes = pool._processes
            chunk_size = X_batch.shape[1] // num_processes
            
            if chunk_size < 1: # Batch muy pequeño, ejecutar secuencial
                 task_args = [(self.W1, self.b1, self.W2, self.b2, X_batch, Y_batch)]
            else:
                task_args = []
                for p in range(num_processes):
                    start_c = p * chunk_size
                    # El último proceso toma lo que sobre
                    end_c = (p + 1) * chunk_size if p < num_processes - 1 else X_batch.shape[1]
                    
                    X_chunk = X_batch[:, start_c:end_c]
                    Y_chunk = Y_batch[:, start_c:end_c]
                    
                    # Empaquetar argumentos (copia pesada de pesos, ¡ojo al overhead!)
                    task_args.append((self.W1, self.b1, self.W2, self.b2, X_chunk, Y_chunk))
            
            # Ejecutar en paralelo (Map)
            results = pool.map(compute_gradients, task_args)
            
            # Agregar (Reduce)
            # Promediar gradientes ponderados por el tamaño del chunk
            total_m = sum(r[4] for r in results)
            
            avg_dW1 = sum(r[0] * (r[4]/total_m) for r in results)
            avg_db1 = sum(r[1] * (r[4]/total_m) for r in results)
            avg_dW2 = sum(r[2] * (r[4]/total_m) for r in results)
            avg_db2 = sum(r[3] * (r[4]/total_m) for r in results)
            
            # Actualizar Pesos
            self.W1 -= self.lr * avg_dW1
            self.b1 -= self.lr * avg_db1
            self.W2 -= self.lr * avg_dW2
            self.b2 -= self.lr * avg_db2