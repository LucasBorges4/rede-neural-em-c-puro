#include "neural-network.h"
#include "activation-function.h"
#include <stdio.h>

int main() {
    int layer_count = 4;
    int neurons_per_layer[] = {4, 4, 4, 4};
    double (*activations[])(double*, int) = {sigmoid, sigmoid, sigmoid, sigmoid};
    NeuralNetwork* nn = create_neural_network(layer_count, neurons_per_layer, activations);

    double input_matrix[4][4] = {
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 1.0},
        {1.0, 0.0, 1.0, 0.0},
        {1.0, 1.0, 1.0, 1.0}
    };

    double expected_outputs[4][4] = {
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 1.0},
        {1.0, 0.0, 1.0, 0.0},
        {1.0, 1.0, 1.0, 1.0}
    };

    int num_examples = 4;
    const int epochs = 1000;
    const double learning_rate = 0.1;

    for (int ep = 0; ep < epochs; ep++) {
        for (int e = 0; e < num_examples; e++) {
            Layer* prev_layer = NULL;
            double* current_input = input_matrix[e];
            double output = 0.0;

            for (int l = 0; l < layer_count; l++) {
                output = forward_propagation(prev_layer, &nn->layers[l], current_input);
                prev_layer = &nn->layers[l];
                current_input = NULL;
            }

            backward_propagation(nn, expected_outputs[e], learning_rate);
        }
    }

    // Teste final
    for (int e = 0; e < num_examples; e++) {
        Layer* prev_layer = NULL;
        double* current_input = input_matrix[e];
        double output = 0.0;

        for (int l = 0; l < layer_count; l++) {
            output = forward_propagation(prev_layer, &nn->layers[l], current_input);
            prev_layer = &nn->layers[l];
            current_input = NULL;
        }
        printf("Exemplo %d -> Saída final: %f\n", e, output);
    }

    free_neural_network(nn);
    return 0;
}

