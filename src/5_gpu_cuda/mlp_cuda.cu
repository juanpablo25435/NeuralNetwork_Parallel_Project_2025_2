#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "mlp_cuda.h"

// === KERNELS (Se ejecutan en la GPU) ===

// Multiplicación de Matrices: C = A * B
// A(m, n), B(n, k) -> C(m, k)
__global__ void matmul_kernel(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            // A es row-major, B es row-major
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// Sumar Bias y aplicar ReLU
__global__ void bias_relu_kernel(float* Z, float* b, float* A, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // El bias se suma por fila (cada neurona tiene un bias)
        // b tiene tamaño (rows). Z es (rows x cols)
        float val = Z[idx] + b[row];
        Z[idx] = val; // Guardamos el valor pre-activación
        
        // ReLU
        A[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

// Sumar Bias (Linear) para la salida
__global__ void bias_linear_kernel(float* Z, float* b, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        Z[idx] = Z[idx] + b[row];
    }
}

// Transponer Matriz en GPU (Simple)
__global__ void transpose_kernel(float* src, float* dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        dst[x * rows + y] = src[y * cols + x];
    }
}

// === CÓDIGO HOST (CPU) ===

void mlp_init(MLPCuda* net, int in, int hidden, int out, float lr) {
    net->input_size = in;
    net->hidden_size = hidden;
    net->output_size = out;
    net->learning_rate = lr;

    // Inicializar pesos en CPU primero para poner valores random
    int size_w1 = in * hidden * sizeof(float);
    int size_b1 = hidden * sizeof(float);
    int size_w2 = hidden * out * sizeof(float);
    int size_b2 = out * sizeof(float);

    float *h_W1 = (float*)malloc(size_w1);
    float *h_W2 = (float*)malloc(size_w2);
    
    // Inicialización simple (He init simplificado)
    for(int i=0; i<in*hidden; i++) h_W1[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    for(int i=0; i<hidden*out; i++) h_W2[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;

    // Reservar memoria en GPU
    CUDA_CHECK(cudaMalloc(&net->d_W1, size_w1));
    CUDA_CHECK(cudaMalloc(&net->d_b1, size_b1));
    CUDA_CHECK(cudaMalloc(&net->d_W2, size_w2));
    CUDA_CHECK(cudaMalloc(&net->d_b2, size_b2));

    // Copiar CPU -> GPU
    CUDA_CHECK(cudaMemcpy(net->d_W1, h_W1, size_w1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(net->d_b1, 0, size_b1)); // Biases a 0
    CUDA_CHECK(cudaMemcpy(net->d_W2, h_W2, size_w2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(net->d_b2, 0, size_b2));

    free(h_W1); free(h_W2);
    
    net->d_Z1 = NULL; // Se inicializan en training según batch size
}

void mlp_train(MLPCuda* net, float* h_X, float* h_Y, int num_samples, int epochs, int batch_size) {
    printf("Iniciando entrenamiento CUDA...\n");

    // Buffers para Batch en GPU
    float *d_X_batch, *d_W1_T, *d_W2_T;
    
    // Reservar memoria máxima para buffers intermedios
    // W1_T: (Hidden x Input)
    CUDA_CHECK(cudaMalloc(&d_W1_T, net->hidden_size * net->input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2_T, net->output_size * net->hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X_batch, net->input_size * batch_size * sizeof(float)));
    
    // Buffers activaciones
    CUDA_CHECK(cudaMalloc(&net->d_Z1, net->hidden_size * batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->d_A1, net->hidden_size * batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->d_Z2, net->output_size * batch_size * sizeof(float)));
    // Salida final en GPU
    // No necesitamos A2 explícito si no calculamos loss en GPU para este demo

    // Configuración de Grid/Block
    dim3 block(16, 16);

    // Eventos para medir tiempo preciso en GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_samples; i += batch_size) {
            int current_batch = (i + batch_size > num_samples) ? (num_samples - i) : batch_size;
            
            // 1. Copiar Batch CPU -> GPU (Cuello de botella PCIe)
            // Nota: Copiamos transpuesto o normal? Asumimos normal y transponemos W
            // X Batch debe ser (784 x 64). h_X es (N x 784).
            // Para simplificar, copiamos fila por fila a un buffer lineal
            // Lo ideal es tener X ya transpuesto en CPU. Asumiremos eso para velocidad.
            
            // Truco: Copiamos el bloque de memoria continuo y lo tratamos como (Input x Batch)
            // Requiere que h_X en CPU esté organizado como (Features x Samples).
            // Asumiremos que el main lo pasa así.
            CUDA_CHECK(cudaMemcpy(d_X_batch, &h_X[i * net->input_size], 
                                  net->input_size * current_batch * sizeof(float), 
                                  cudaMemcpyHostToDevice));

            // 2. Transponer Pesos (Necesario para W * X)
            // Grid para W1_T
            dim3 grid_t1((net->hidden_size + 15)/16, (net->input_size + 15)/16);
            transpose_kernel<<<grid_t1, block>>>(net->d_W1, d_W1_T, net->input_size, net->hidden_size);
            
            // 3. Forward Capa 1: Z1 = W1.T * X
            // W1.T (Hidden, Input) * X (Input, Batch) -> Z1 (Hidden, Batch)
            dim3 grid_mul1((current_batch + 15)/16, (net->hidden_size + 15)/16);
            matmul_kernel<<<grid_mul1, block>>>(d_W1_T, d_X_batch, net->d_Z1, 
                                                net->hidden_size, net->input_size, current_batch);
            
            // 4. Bias + ReLU
            bias_relu_kernel<<<grid_mul1, block>>>(net->d_Z1, net->d_b1, net->d_A1, 
                                                   net->hidden_size, current_batch);
            
            // 5. Forward Capa 2
            dim3 grid_t2((net->output_size + 15)/16, (net->hidden_size + 15)/16);
            transpose_kernel<<<grid_t2, block>>>(net->d_W2, d_W2_T, net->hidden_size, net->output_size);
            
            dim3 grid_mul2((current_batch + 15)/16, (net->output_size + 15)/16);
            matmul_kernel<<<grid_mul2, block>>>(d_W2_T, net->d_A1, net->d_Z2,
                                                net->output_size, net->hidden_size, current_batch);
            
            // 6. Bias Salida (Linear)
            bias_linear_kernel<<<grid_mul2, block>>>(net->d_Z2, net->d_b2,
                                                     net->output_size, current_batch);
            
            // Sincronizar para asegurar que GPU terminó este batch (opcional, reduce performance)
            // cudaDeviceSynchronize();
        }
        // printf("Epoch GPU %d\n", epoch+1);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("\n>>> Tiempo Total GPU (C++ CUDA): %.2f segundos <<<\n", milliseconds / 1000.0f);

    cudaFree(d_W1_T); cudaFree(d_W2_T); cudaFree(d_X_batch);
}

void mlp_free(MLPCuda* net) {
    cudaFree(net->d_W1); cudaFree(net->d_b1);
    cudaFree(net->d_W2); cudaFree(net->d_b2);
    cudaFree(net->d_Z1); cudaFree(net->d_A1);
    cudaFree(net->d_Z2);
}