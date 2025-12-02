#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdint.h>

// Lee imágenes MNIST y devuelve array de floats normalizados (0.0-1.0)
// Los valores de rows, cols y num_images se llenan por referencia
float* load_mnist_images(const char* filename, int* num_images, int* rows, int* cols);

// Lee etiquetas MNIST
uint8_t* load_mnist_labels(const char* filename, int* num_labels);

void free_mnist_data(float* images, uint8_t* labels);

#endif