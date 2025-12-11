#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "../common/mnist_loader.h"
#include "mlp.h"

// Rutas a los datos DESCOMPRIMIDOS
#define TRAIN_IMG "../../data/train-images-idx3-ubyte"
#define TRAIN_LBL "../../data/train-labels-idx1-ubyte"
#define TEST_IMG  "../../data/t10k-images-idx3-ubyte"
#define TEST_LBL  "../../data/t10k-labels-idx1-ubyte"

int main() {
    printf("--- Escenario 1b: C Secuencial Puro ---\n");
    
    int num_train, r, c, num_lbl;
    float* X_train = load_mnist_images(TRAIN_IMG, &num_train, &r, &c);
    uint8_t* Y_train = load_mnist_labels(TRAIN_LBL, &num_lbl);
    
    if (!X_train || !Y_train) {
        printf("ERROR: No se encontraron los datos. Ejecuta 'gunzip -k data/*.gz'\n");
        return 1;
    }
    
    printf("Datos cargados: %d imagenes (%dx%d)\n", num_train, r, c);
    
    MLP mlp;
    mlp_init(&mlp, 784, 512, 10, 0.1f);
    
    int EPOCHS = 10;
    int BATCH_SIZE = 64;
    
    // Buffers para el batch
    float* X_batch = malloc(784 * BATCH_SIZE * sizeof(float));
    float* Y_batch_enc = calloc(10 * BATCH_SIZE, sizeof(float)); // One-hot
    
    clock_t start = clock();
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < num_train; i += BATCH_SIZE) {
            int current_batch = (i + BATCH_SIZE > num_train) ? (num_train - i) : BATCH_SIZE;
            
            // Preparar Batch
            // X_train[img][pixel] -> X_batch[pixel][img]
            memset(Y_batch_enc, 0, 10 * BATCH_SIZE * sizeof(float));
            
            for (int b = 0; b < current_batch; b++) {
                int img_idx = i + b;
                for (int p = 0; p < 784; p++) {
                    X_batch[p * BATCH_SIZE + b] = X_train[img_idx * 784 + p];
                }
                // One Hot
                int label = Y_train[img_idx];
                Y_batch_enc[label * BATCH_SIZE + b] = 1.0f;
            }
            
            mlp_forward(&mlp, X_batch, BATCH_SIZE);
            mlp_backward(&mlp, X_batch, Y_batch_enc, BATCH_SIZE);
        }
        printf("Epoch %d terminada.\n", epoch + 1);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("\n>>> Tiempo Total Entrenamiento (C): %.2f segundos <<<\n", time_spent);
    
    free(X_train); free(Y_train); free(X_batch); free(Y_batch_enc);
    mlp_free(&mlp);
    return 0;
}