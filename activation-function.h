#include <stdio.h>
#include <math.h>


// Sigmoid activation function 
double sigmoid(double* x);
int relu(int* x);
int tanh_activation(int* x);
int linear(int* x);

// Derivative of activation functions
int sigmoid_derivative(int* x);
int relu_derivative(int* x);
int tanh_derivative(int* x);
int linear_derivative(int* x);
