#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "mlp.h"

float rand_weight_omp() { return ((float)rand() / RAND_MAX - 0.5f) * 0.1f; }

void mlp_init(MLP* net, int in, int hidden, int out, float lr) {
    net->input_size = in;
    net->hidden_size = hidden;
    net->output_size = out;
    net->learning_rate = lr;
    
    net->W1 = malloc(in * hidden * sizeof(float));
    net->b1 = calloc(hidden, sizeof(float));
    net->W2 = malloc(hidden * out * sizeof(float));
    net->b2 = calloc(out, sizeof(float));
    
    for(int i=0; i<in*hidden; i++) net->W1[i] = rand_weight_omp();
    for(int i=0; i<hidden*out; i++) net->W2[i] = rand_weight_omp();
    
    net->Z1 = NULL; net->A1 = NULL;
    net->Z2 = NULL; net->A2 = NULL;
}

// === PARALELISMO ===
void mat_mul(float* A, float* B, float* C, int m, int n, int p) {
    memset(C, 0, m * p * sizeof(float));
    
    // Paralelizamos el bucle externo (filas de A / filas de C)
    // schedule(static) reparte las filas equitativamente entre los hilos
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            float a_val = A[i * n + k];
            // Este bucle interno queda serial para que el hilo 'i' complete su fila
            for (int j = 0; j < p; j++) {
                C[i * p + j] += a_val * B[k * p + j];
            }
        }
    }
}

void mat_transpose(float* src, float* dst, int rows, int cols) {
    #pragma omp parallel for schedule(static)
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

void mlp_forward(MLP* net, float* X, int batch_size) {
    if (!net->Z1) {
        net->Z1 = malloc(net->hidden_size * batch_size * sizeof(float));
        net->A1 = malloc(net->hidden_size * batch_size * sizeof(float));
        net->Z2 = malloc(net->output_size * batch_size * sizeof(float));
        net->A2 = malloc(net->output_size * batch_size * sizeof(float));
    }
    
    // 1. Capa Oculta
    float* W1_T = malloc(net->hidden_size * net->input_size * sizeof(float));
    mat_transpose(net->W1, W1_T, net->input_size, net->hidden_size);
    
    // mat_mul ya tiene #pragma adentro, así que se acelera automáticamente
    mat_mul(W1_T, X, net->Z1, net->hidden_size, net->input_size, batch_size);
    free(W1_T);
    
    // Bias + ReLU (Paralelizable)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < batch_size; j++) {
            int idx = i * batch_size + j;
            net->Z1[idx] += net->b1[i];
            net->A1[idx] = (net->Z1[idx] > 0) ? net->Z1[idx] : 0.0f;
        }
    }
    
    // 2. Capa Salida
    float* W2_T = malloc(net->output_size * net->hidden_size * sizeof(float));
    mat_transpose(net->W2, W2_T, net->hidden_size, net->output_size);
    mat_mul(W2_T, net->A1, net->Z2, net->output_size, net->hidden_size, batch_size);
    free(W2_T);
    
    // Softmax - Paralelizamos por ejemplo la columna j
    #pragma omp parallel for
    for (int j = 0; j < batch_size; j++) {
        float max_val = -1e9;
        for (int i = 0; i < net->output_size; i++) {
            if (net->Z2[i * batch_size + j] > max_val) 
                max_val = net->Z2[i * batch_size + j];
        }
        
        float sum = 0.0f;
        for (int i = 0; i < net->output_size; i++) {
            int idx = i * batch_size + j;
            float z = net->Z2[idx] + net->b2[i]; 
            net->Z2[idx] = z;
            float val = expf(z - max_val);
            net->A2[idx] = val;
            sum += val;
        }
        
        for (int i = 0; i < net->output_size; i++) {
            net->A2[i * batch_size + j] /= sum;
        }
    }
}

void mlp_backward(MLP* net, float* X, float* Y, int batch_size) {
    // dZ2
    float* dZ2 = malloc(net->output_size * batch_size * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < net->output_size * batch_size; i++) {
        dZ2[i] = net->A2[i] - Y[i];
    }
    
    // dW2
    float* dZ2_T = malloc(batch_size * net->output_size * sizeof(float));
    mat_transpose(dZ2, dZ2_T, net->output_size, batch_size);
    
    float* dW2 = malloc(net->hidden_size * net->output_size * sizeof(float));
    mat_mul(net->A1, dZ2_T, dW2, net->hidden_size, batch_size, net->output_size);
    free(dZ2_T);
    
    // db2 - Suma por filas. Paralelizamos el bucle externo
    float* db2 = calloc(net->output_size, sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < net->output_size; i++) {
        for (int j = 0; j < batch_size; j++) {
            db2[i] += dZ2[i * batch_size + j];
        }
    }
    
    // dA1
    float* dA1 = malloc(net->hidden_size * batch_size * sizeof(float));
    mat_mul(net->W2, dZ2, dA1, net->hidden_size, net->output_size, batch_size);
    
    // dZ1
    float* dZ1 = malloc(net->hidden_size * batch_size * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < net->hidden_size * batch_size; i++) {
        dZ1[i] = (net->Z1[i] > 0) ? dA1[i] : 0.0f;
    }
    
    // dW1
    float* dZ1_T = malloc(batch_size * net->hidden_size * sizeof(float));
    mat_transpose(dZ1, dZ1_T, net->hidden_size, batch_size);
    
    float* dW1 = malloc(net->input_size * net->hidden_size * sizeof(float));
    mat_mul(X, dZ1_T, dW1, net->input_size, batch_size, net->hidden_size);
    free(dZ1_T);
    
    // db1
    float* db1 = calloc(net->hidden_size, sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < batch_size; j++) {
            db1[i] += dZ1[i * batch_size + j];
        }
    }
    
    // Update Weights - Paralelización masiva
    float inv_m = 1.0f / batch_size;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Actualizar W1
            #pragma omp parallel for
            for (int i = 0; i < net->input_size * net->hidden_size; i++) 
                net->W1[i] -= net->learning_rate * dW1[i] * inv_m;
        }
        
        #pragma omp section
        {
            // Actualizar b1
            #pragma omp parallel for
            for (int i = 0; i < net->hidden_size; i++) 
                net->b1[i] -= net->learning_rate * db1[i] * inv_m;
        }
        
        #pragma omp section
        {
            // Actualizar W2
            #pragma omp parallel for
            for (int i = 0; i < net->hidden_size * net->output_size; i++) 
                net->W2[i] -= net->learning_rate * dW2[i] * inv_m;
        }
        
        #pragma omp section
        {
            // Actualizar b2
            #pragma omp parallel for
            for (int i = 0; i < net->output_size; i++) 
                net->b2[i] -= net->learning_rate * db2[i] * inv_m;
        }
    }

    // Limpieza
    free(dZ2); free(dW2); free(db2);
    free(dA1); free(dZ1); free(dW1); free(db1);
}

void mlp_free(MLP* net) {
    free(net->W1); free(net->b1);
    free(net->W2); free(net->b2);
    if(net->Z1) { free(net->Z1); free(net->A1); free(net->Z2); free(net->A2); }
}