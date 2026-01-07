#include "neural-network.h"
#include "activation-function.h"
#include <stdio.h>

int main() {
    int layer_count = 4;
    int neurons_per_layer[] = {4, 4, 4, 4};
    double (*activations[])(double*, int) = {sigmoid, sigmoid, sigmoid, sigmoid};
    NeuralNetwork* nn = create_neural_network(layer_count, neurons_per_layer, activations);

    // -------------------- Caso 1: vetor simples --------------------
    double input_simple[] = {0.5, 5, 9.8, 0.3}; // 4 features

    Layer* prev_layer = NULL;
    double* current_input = input_simple;
    double output = 0.0;

    for (int l = 0; l < layer_count; l++) {
        output = forward_propagation(prev_layer, &nn->layers[l], current_input);
        prev_layer = &nn->layers[l];
        current_input = NULL;
    }
    double vetor_simple_output = output;

    // -------------------- Caso 2: matriz de exemplos --------------------
    double input_matrix[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    int num_examples = 4;
    for (int e = 0; e < num_examples; e++) {
        prev_layer = NULL;
        current_input = input_matrix[e]; // pega a linha da matriz
        output = 0.0;

        for (int l = 0; l < layer_count; l++) {
            output = forward_propagation(prev_layer, &nn->layers[l], current_input);
            prev_layer = &nn->layers[l];
            current_input = NULL;
        }
        printf("Exemplo %d -> Saída da rede: %f\n", e, output);
    }
    double vetor_composto_output = output;

    // Debug
    print_neural_network(nn);

    printf("Vetor simples -> Saída da rede: %f\n", vetor_simple_output);
    printf("Vetor composto -> Saída da rede: %f\n", vetor_composto_output);

    free_neural_network(nn);
    return 0;
}
