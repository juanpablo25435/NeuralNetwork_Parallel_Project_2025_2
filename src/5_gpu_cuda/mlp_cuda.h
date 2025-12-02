#ifndef MLP_CUDA_H
#define MLP_CUDA_H

#include <cuda_runtime.h>

// Macro para chequear errores de CUDA (Buenas prácticas)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s en %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

// Estructura de la red
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    float learning_rate;

    // Punteros en el DEVICE (GPU)
    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    
    // Buffers intermedios en DEVICE
    float *d_Z1, *d_A1;
    float *d_Z2, *d_A2;
} MLPCuda;

// Funciones
void mlp_init(MLPCuda* net, int in, int hidden, int out, float lr);
void mlp_train(MLPCuda* net, float* h_X, float* h_Y, int num_samples, int epochs, int batch_size);
void mlp_free(MLPCuda* net);

#endif