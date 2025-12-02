#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h> // Para imprimir número de hilos
#include "../common/mnist_loader.h"
#include "mlp.h"

// Rutas
#define TRAIN_IMG "../../data/train-images-idx3-ubyte"
#define TRAIN_LBL "../../data/train-labels-idx1-ubyte"

int main() {
    // Verificar OpenMP
    int max_threads = omp_get_max_threads();
    printf("--- Escenario 2b: Paralelo OpenMP ---\n");
    printf(">>> Usando %d hilos de CPU <<<\n", max_threads);
    
    int num_train, r, c, num_lbl;
    float* X_train = load_mnist_images(TRAIN_IMG, &num_train, &r, &c);
    uint8_t* Y_train = load_mnist_labels(TRAIN_LBL, &num_lbl);
    
    if (!X_train || !Y_train) {
        printf("Error cargando datos. Verifica paths.\n");
        return 1;
    }
    
    MLP mlp;
    mlp_init(&mlp, 784, 512, 10, 0.1f);
    
    int EPOCHS = 10;
    int BATCH_SIZE = 64;
    
    float* X_batch = malloc(784 * BATCH_SIZE * sizeof(float));
    float* Y_batch_enc = calloc(10 * BATCH_SIZE, sizeof(float));
    
    printf("Iniciando entrenamiento Paralelo...\n");
    double start_time = omp_get_wtime(); // Usamos timer de OpenMP, más preciso
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < num_train; i += BATCH_SIZE) {
            int current_batch = (i + BATCH_SIZE > num_train) ? (num_train - i) : BATCH_SIZE;
            
            // Preparar Batch (Paralelizable también)
            memset(Y_batch_enc, 0, 10 * BATCH_SIZE * sizeof(float));
            
            #pragma omp parallel for
            for (int b = 0; b < current_batch; b++) {
                int img_idx = i + b;
                // Copia paralela
                for (int p = 0; p < 784; p++) {
                    X_batch[p * BATCH_SIZE + b] = X_train[img_idx * 784 + p];
                }
                Y_batch_enc[Y_train[img_idx] * BATCH_SIZE + b] = 1.0f;
            }
            
            mlp_forward(&mlp, X_batch, BATCH_SIZE);
            mlp_backward(&mlp, X_batch, Y_batch_enc, BATCH_SIZE);
        }
        printf("Epoch %d terminada.\n", epoch + 1);
    }
    
    double end_time = omp_get_wtime();
    printf("\n>>> Tiempo Total OpenMP: %.2f segundos <<<\n", end_time - start_time);
    
    free(X_train); free(Y_train); free(X_batch); free(Y_batch_enc);
    mlp_free(&mlp);
    return 0;
}