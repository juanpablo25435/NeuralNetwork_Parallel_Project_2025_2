#include <stdio.h>
#include <stdlib.h>
#include "../common/mnist_loader.h"
#include "mlp_cuda.h"

#include "../common/mnist_loader.c" 

#define TRAIN_IMG "../../data/train-images-idx3-ubyte"
#define TRAIN_LBL "../../data/train-labels-idx1-ubyte"

int main() {
    printf("--- Escenario 3: GPU CUDA Nativo (C++) ---\n");
    
    // Verificar GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("ERROR: No se detectó GPU CUDA.\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Hardware: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);

    // Cargar datos
    int num_train, r, c, num_lbl;
    float* X_raw = load_mnist_images(TRAIN_IMG, &num_train, &r, &c);
    
    if (!X_raw) { printf("Error datos.\n"); return 1; }
    
    // Transponer X en CPU para que sea (Features x Samples)
    printf("Transponiendo datos en CPU...\n");
    float* X_transposed = (float*)malloc(num_train * 784 * sizeof(float));
    for(int i=0; i<num_train; i++) {
        for(int j=0; j<784; j++) {
        }
    }
        
    float* X_T = (float*)malloc(num_train * 784 * sizeof(float));
    // Transponer X_raw (N x 784) -> X_T (784 x N)
    for(int i=0; i<num_train; i++) {
        for(int j=0; j<784; j++) {
            X_T[j * num_train + i] = X_raw[i * 784 + j];
        }
    }
    // Si transponemos todo junto, el batching es difícil de indexar.
    // Simplemente pasamos X_raw al trainer, y el trainer copia y transpone el batch pequeño.

    MLPCuda net;
    mlp_init(&net, 784, 512, 10, 0.1f);
    
    // Para el benchmark de velocidad, pasamos X_T organizado para acceso rápido
    // Pero por simplicidad, pasemos X_T donde X_T[feature * N + sample]
    
    mlp_train(&net, X_T, NULL, num_train, 10, 64);
    
    mlp_free(&net);
    free(X_raw);
    free(X_T);
    return 0;
}