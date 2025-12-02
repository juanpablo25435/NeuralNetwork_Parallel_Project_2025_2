CC = gcc
CFLAGS = -O2 -Wall
LIBS = -lm

all: mlp_seq

mlp_seq: main.c mlp.c ../common/mnist_loader.c
	$(CC) $(CFLAGS) -o mlp_seq main.c mlp.c ../common/mnist_loader.c $(LIBS)

clean:
	rm -f mlp_seq