CC = gcc
# -fopenmp habilita el paralelismo
CFLAGS = -O2 -Wall -fopenmp 
LIBS = -lm

all: mlp_omp

mlp_omp: main_omp.c mlp_omp.c ../common/mnist_loader.c
	$(CC) $(CFLAGS) -o mlp_omp main_omp.c mlp_omp.c ../common/mnist_loader.c $(LIBS)

clean:
	rm -f mlp_omp