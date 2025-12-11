#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mlp.h"

// Inicialización random simple
float rand_weight() { return ((float)rand() / RAND_MAX - 0.5f) * 0.1f; }

void mlp_init(MLP* net, int in, int hidden, int out, float lr) {
    net->input_size = in;
    net->hidden_size = hidden;
    net->output_size = out;
    net->learning_rate = lr;

    net->W1 = malloc(in * hidden * sizeof(float));
    net->b1 = calloc(hidden, sizeof(float));
    net->W2 = malloc(hidden * out * sizeof(float));
    net->b2 = calloc(out, sizeof(float));
    
    // Inicializar W1 y W2
    for(int i=0; i<in*hidden; i++) net->W1[i] = rand_weight();
    for(int i=0; i<hidden*out; i++) net->W2[i] = rand_weight();
    
    // Punteros a NULL para allocar luego dinámicamente según batch size
    net->Z1 = NULL; net->A1 = NULL;
    net->Z2 = NULL; net->A2 = NULL;
}

// Multiplicación de matrices C = A * B
// A: (m x n), B: (n x p) -> C: (m x p)
void mat_mul(float* A, float* B, float* C, int m, int n, int p) {
    // Limpiar C
    memset(C, 0, m * p * sizeof(float));
    
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            float a_val = A[i * n + k];
            for (int j = 0; j < p; j++) {
                C[i * p + j] += a_val * B[k * p + j];
            }
        }
    }
}

// Transponer matriz (necesario para backprop)
void mat_transpose(float* src, float* dst, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

void mlp_forward(MLP* net, float* X, int batch_size) {
    // X entra como (Input x Batch)
    
    // Reservar memoria temporal si cambia el batch_size
    if (!net->Z1) {
        net->Z1 = malloc(net->hidden_size * batch_size * sizeof(float));
        net->A1 = malloc(net->hidden_size * batch_size * sizeof(float));
        net->Z2 = malloc(net->output_size * batch_size * sizeof(float));
        net->A2 = malloc(net->output_size * batch_size * sizeof(float));
    }
    
    // 1. Z1 = W1.T * X + b1
    // W1 es (In x Hidden). W1.T es (Hidden x In).
    // X es (In x Batch). Resultado (Hidden x Batch).
    
    // Transponer W1
    float* W1_T = malloc(net->hidden_size * net->input_size * sizeof(float));
    mat_transpose(net->W1, W1_T, net->input_size, net->hidden_size);
    
    mat_mul(W1_T, X, net->Z1, net->hidden_size, net->input_size, batch_size);
    free(W1_T);
    
    // Sumar bias y ReLU
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < batch_size; j++) {
            int idx = i * batch_size + j;
            net->Z1[idx] += net->b1[i]; // Broadcast bias
            // ReLU
            net->A1[idx] = (net->Z1[idx] > 0) ? net->Z1[idx] : 0.0f;
        }
    }
    
    // 2. Z2 = W2.T * A1 + b2
    // W2 (Hidden x Out) -> W2.T (Out x Hidden)
    float* W2_T = malloc(net->output_size * net->hidden_size * sizeof(float));
    mat_transpose(net->W2, W2_T, net->hidden_size, net->output_size);
    
    mat_mul(W2_T, net->A1, net->Z2, net->output_size, net->hidden_size, batch_size);
    free(W2_T);
    
    // Sumar bias y Softmax
    for (int j = 0; j < batch_size; j++) {
        // Encontrar max para estabilidad numérica
        float max_val = -1e9;
        for (int i = 0; i < net->output_size; i++) {
            if (net->Z2[i * batch_size + j] > max_val) 
                max_val = net->Z2[i * batch_size + j];
        }
        
        float sum = 0.0f;
        for (int i = 0; i < net->output_size; i++) {
            int idx = i * batch_size + j;
            // Sumar bias antes de exp
            float z = net->Z2[idx] + net->b2[i]; 
            net->Z2[idx] = z; // Actualizar Z2 con bias incluido
            
            float val = expf(z - max_val); // Softmax estable
            net->A2[idx] = val;
            sum += val;
        }
        
        for (int i = 0; i < net->output_size; i++) {
            net->A2[i * batch_size + j] /= sum;
        }
    }
}

void mlp_backward(MLP* net, float* X, float* Y, int batch_size) {
    // dZ2 = A2 - Y
    float* dZ2 = malloc(net->output_size * batch_size * sizeof(float));
    for (int i = 0; i < net->output_size * batch_size; i++) {
        dZ2[i] = net->A2[i] - Y[i];
    }
    
    // dW2 = A1 * dZ2.T / m
    // A1 (Hidden x Batch), dZ2 (Out x Batch) -> dZ2.T (Batch x Out)
    // Res: (Hidden x Out) que coincide con W2
    float* dZ2_T = malloc(batch_size * net->output_size * sizeof(float));
    mat_transpose(dZ2, dZ2_T, net->output_size, batch_size);
    
    float* dW2 = malloc(net->hidden_size * net->output_size * sizeof(float));
    mat_mul(net->A1, dZ2_T, dW2, net->hidden_size, batch_size, net->output_size);
    
    // db2 = sum(dZ2)
    float* db2 = calloc(net->output_size, sizeof(float));
    for (int i = 0; i < net->output_size; i++) {
        for (int j = 0; j < batch_size; j++) {
            db2[i] += dZ2[i * batch_size + j];
        }
    }
    
    // Backprop a oculta
    // dA1 = W2 * dZ2
    // W2 (Hidden x Out), dZ2 (Out x Batch) -> (Hidden x Batch)
    float* dA1 = malloc(net->hidden_size * batch_size * sizeof(float));
    mat_mul(net->W2, dZ2, dA1, net->hidden_size, net->output_size, batch_size);
    
    // dZ1 = dA1 * ReLU_deriv(Z1)
    float* dZ1 = malloc(net->hidden_size * batch_size * sizeof(float));
    for (int i = 0; i < net->hidden_size * batch_size; i++) {
        dZ1[i] = (net->Z1[i] > 0) ? dA1[i] : 0.0f;
    }
    
    // dW1 = X * dZ1.T / m
    // X (In x Batch), dZ1 (Hidden x Batch) -> dZ1.T (Batch x Hidden)
    // Res: (In x Hidden)
    float* dZ1_T = malloc(batch_size * net->hidden_size * sizeof(float));
    mat_transpose(dZ1, dZ1_T, net->hidden_size, batch_size);
    
    float* dW1 = malloc(net->input_size * net->hidden_size * sizeof(float));
    mat_mul(X, dZ1_T, dW1, net->input_size, batch_size, net->hidden_size);
    
    // db1
    float* db1 = calloc(net->hidden_size, sizeof(float));
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < batch_size; j++) {
            db1[i] += dZ1[i * batch_size + j];
        }
    }
    
    // Update Weights
    float inv_m = 1.0f / batch_size;
    for (int i = 0; i < net->input_size * net->hidden_size; i++) 
        net->W1[i] -= net->learning_rate * dW1[i] * inv_m;
        
    for (int i = 0; i < net->hidden_size; i++) 
        net->b1[i] -= net->learning_rate * db1[i] * inv_m;
        
    for (int i = 0; i < net->hidden_size * net->output_size; i++) 
        net->W2[i] -= net->learning_rate * dW2[i] * inv_m;
        
    for (int i = 0; i < net->output_size; i++) 
        net->b2[i] -= net->learning_rate * db2[i] * inv_m;

    // Limpieza
    free(dZ2); free(dZ2_T); free(dW2); free(db2);
    free(dA1); free(dZ1); free(dZ1_T); free(dW1); free(db1);
}

void mlp_free(MLP* net) {
    free(net->W1); free(net->b1);
    free(net->W2); free(net->b2);
    if(net->Z1) { free(net->Z1); free(net->A1); free(net->Z2); free(net->A2); }
}