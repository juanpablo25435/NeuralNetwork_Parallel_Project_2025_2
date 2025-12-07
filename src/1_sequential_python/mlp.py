import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Inicialización de Pesos y Sesgos
        # Según PDF 4.1: W[1] es (784 x N) y b[1] es (N x 1) [cite: 58]
        # Usamos np.random.randn * 0.01 para inicialización pequeña aleatoria
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        
        # W[2] es (N x 10) y b[2] es (10 x 1) [cite: 59]
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def relu(self, Z):
        # Fase 1: Activación ReLU [cite: 66]
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        # Derivada de ReLU para Backprop [cite: 89]
        return Z > 0

    def softmax(self, Z):
        # Fase 1: Activación Softmax [cite: 70]
        # Restamos max(Z) por estabilidad numérica (evita overflow de exp)
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward(self, X):
        # X shape: (784, batch_size)
        
        # 1. Cálculo Capa Oculta (Pre-activación) [cite: 64]
        # Z1 = W1.T . X + b1
        self.Z1 = np.dot(self.W1.T, X) + self.b1
        
        # 2. Activación Capa Oculta [cite: 66]
        self.A1 = self.relu(self.Z1)
        
        # 3. Cálculo Capa Salida [cite: 68]
        # Z2 = W2.T . A1 + b2
        self.Z2 = np.dot(self.W2.T, self.A1) + self.b2
        
        # 4. Activación Capa Salida [cite: 70]
        self.A2 = self.softmax(self.Z2)
        
        return self.A2

    def backward(self, X, Y):
        # Propagación Hacia Atrás [cite: 83-91]
        m = X.shape[1] # Tamaño del batch
        
        # dz2 = A2 - Y [cite: 85]
        dZ2 = self.A2 - Y
        
        # dW2 = A1 . dz2.T (Promediado por m) [cite: 86]
        # Nota: Ajustamos dimensiones para batch: dW2 = dot(A1, dZ2.T)
        dW2 = np.dot(self.A1, dZ2.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m # [cite: 87]
        
        # Backprop a capa oculta
        # da1 = W2 . dz2 [cite: 88]
        dA1 = np.dot(self.W2, dZ2)
        
        # dz1 = da1 * ReLU'(z1) [cite: 89]
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        
        # dW1 = X . dz1.T [cite: 90]
        dW1 = np.dot(X, dZ1.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m # [cite: 91]
        
        # Fase 4: Actualización de Pesos [cite: 94-98]
        self.W1 = self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr * db1
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2