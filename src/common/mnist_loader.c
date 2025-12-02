#include <stdio.h>
#include <stdlib.h>
#include "mnist_loader.h"

// Función para invertir bytes (Big Endian -> Little Endian)
uint32_t flip_bytes(uint32_t v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
           ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000);
}

float* load_mnist_images(const char* filename, int* num_images, int* rows, int* cols) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: No se pudo abrir %s\n", filename);
        return NULL;
    }

    uint32_t magic, n, r, c;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&n, sizeof(uint32_t), 1, f);
    fread(&r, sizeof(uint32_t), 1, f);
    fread(&c, sizeof(uint32_t), 1, f);

    *num_images = flip_bytes(n);
    *rows = flip_bytes(r);
    *cols = flip_bytes(c);

    int total_pixels = (*num_images) * (*rows) * (*cols);
    
    // Leer datos crudos (bytes)
    uint8_t* raw_data = (uint8_t*)malloc(total_pixels * sizeof(uint8_t));
    fread(raw_data, sizeof(uint8_t), total_pixels, f);
    fclose(f);

    // Convertir a float normalizado
    float* data = (float*)malloc(total_pixels * sizeof(float));
    for (int i = 0; i < total_pixels; i++) {
        data[i] = (float)raw_data[i] / 255.0f;
    }
    
    free(raw_data);
    return data;
}

uint8_t* load_mnist_labels(const char* filename, int* num_labels) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    uint32_t magic, n;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&n, sizeof(uint32_t), 1, f);

    *num_labels = flip_bytes(n);

    uint8_t* labels = (uint8_t*)malloc(*num_labels * sizeof(uint8_t));
    fread(labels, sizeof(uint8_t), *num_labels, f);
    fclose(f);

    return labels;
}

void free_mnist_data(float* images, uint8_t* labels) {
    if (images) free(images);
    if (labels) free(labels);
}