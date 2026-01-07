#include <stdio.h>
#include <math.h>

// Sigmoid activation function
double sigmoid(double *x, int dimension) {
    double applied_value = 0.0;
    for (int i = 0; i < dimension; i++) {
        applied_value += 1 / (1 + exp(-x[i]));
    }
    return applied_value;
}


int relu(int* x);
int tanh_activation(int* x);
int linear(int* x);
