#ifndef MLP_H
#define MLP_H

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    float learning_rate;
    
    // Pesos y Sesgos (Arrays planos)
    float *W1, *b1; // W1: input x hidden
    float *W2, *b2; // W2: hidden x output
    
    // Activaciones intermedias (para Backpropagation)
    float *Z1, *A1;
    float *Z2, *A2;
} MLP;

void mlp_init(MLP* mlp, int in, int hidden, int out, float lr);
void mlp_forward(MLP* mlp, float* X, int batch_size);
void mlp_backward(MLP* mlp, float* X, float* Y, int batch_size);
void mlp_free(MLP* mlp);

#endif