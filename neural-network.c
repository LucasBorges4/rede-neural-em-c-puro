#include "neural-network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// -------------------- Utilitários --------------------

double random_weight() {
    static int initialized = 0;
    if (!initialized) {
        srand((unsigned int) time(NULL));
        initialized = 1;
    }
    // peso aleatório entre -1 e 1
    return ((double) rand() / RAND_MAX) * 2.0 - 1.0;
}

// -------------------- Inicialização --------------------

void initialize_weights(Weight* weights, int count) {
    for (int i = 0; i < count; i++) {
        weights[i].slope = random_weight();
        weights[i].bias  = random_weight();
    }
}

void initialize_neurons(Neuron* neurons, int neuron_count, int input_count,
                        double (*activation_function)(double*, int)) {
    for (int i = 0; i < neuron_count; i++) {
        neurons[i].weights = (Weight*) malloc(input_count * sizeof(Weight));
        neurons[i].weight_count = input_count;
        neurons[i].activation_function = activation_function;
        neurons[i].output = 0.0;
        initialize_weights(neurons[i].weights, input_count);
    }
}

void initialize_layer(Layer* layer, int neuron_count, int input_count,
                      double (*activation_function)(double*, int)) {
    layer->neurons = (Neuron*) malloc(neuron_count * sizeof(Neuron));
    layer->neuron_count = neuron_count;
    for (int i = 0; i < neuron_count; i++) {
        initialize_neurons(&layer->neurons[i], 1, input_count, activation_function);
    }
}


void initialize_neural_network(NeuralNetwork* nn, int layer_count, int* neurons_per_layer,
                               double (**activation_functions)(double*, int)) {
    nn->layers = (Layer*) malloc(layer_count * sizeof(Layer));
    nn->layer_count = layer_count;

    for (int l = 0; l < layer_count; l++) {
        int input_count = (l == 0) ? neurons_per_layer[l] : nn->layers[l - 1].neuron_count;
        initialize_layer(&nn->layers[l], neurons_per_layer[l], input_count,
                         activation_functions[l]); // one per layer
    }
}


// -------------------- Criação e Liberação --------------------

Neuron* create_neuron(int input_count, double (*activation_function)(double*, int)) {
    Neuron* neuron = (Neuron*) malloc(sizeof(Neuron));
    neuron->weights = (Weight*) malloc(input_count * sizeof(Weight));
    neuron->weight_count = input_count;
    neuron->activation_function = activation_function;
    neuron->output = 0.0;
    initialize_weights(neuron->weights, input_count);
    return neuron;
}

void free_neuron(Neuron* neuron) {
    if (neuron) {
        free(neuron->weights);
        free(neuron);
    }
}

Layer* create_layer(int neuron_count, int input_count,
                    double (**activation_functions)(double*, int)) {
    Layer* layer = (Layer*) malloc(sizeof(Layer));
    layer->neurons = (Neuron*) malloc(neuron_count * sizeof(Neuron));
    layer->neuron_count = neuron_count;
    for (int i = 0; i < neuron_count; i++) {
        initialize_neurons(&layer->neurons[i], 1, input_count, activation_functions[i]);
    }
    return layer;
}

void free_layer(Layer* layer) {
    if (layer) {
        for (int i = 0; i < layer->neuron_count; i++) {
            free(layer->neurons[i].weights);
        }
        free(layer->neurons);
        free(layer);
    }
}

NeuralNetwork* create_neural_network(int layer_count, int* neurons_per_layer,
                                     double (**activation_functions)(double*, int)) {
    NeuralNetwork* nn = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    initialize_neural_network(nn, layer_count, neurons_per_layer, activation_functions);
    return nn;
}

void free_neural_network(NeuralNetwork* nn) {
    if (nn) {
        for (int l = 0; l < nn->layer_count; l++) {
            Layer* layer = &nn->layers[l];
            for (int i = 0; i < layer->neuron_count; i++) {
                free(layer->neurons[i].weights);
            }
            free(layer->neurons);
        }
        free(nn->layers);
        free(nn);
    }
}

// -------------------- Forward Propagation --------------------

double forward_propagation(Layer* prev_layer, Layer* next_layer, double* input) {
    for (int j = 0; j < next_layer->neuron_count; j++) {
        Neuron* neuron = &next_layer->neurons[j];
        double sum = 0.0;
        for (int i = 0; i < neuron->weight_count; i++) {
            double in_val = (prev_layer == NULL) ? input[i] : prev_layer->neurons[i].output;
            // Nota: bias por entrada (design atual). O convencional é bias único por neurônio.
            sum += neuron->weights[i].slope * in_val + neuron->weights[i].bias;
        }
        neuron->output = neuron->activation_function(&sum, 1);
    }
    return next_layer->neurons[next_layer->neuron_count - 1].output;
}

// -------------------- Debug --------------------

void print_neuron(const Neuron* neuron) {
    printf("Neuron: weights=%d, output=%f\n", neuron->weight_count, neuron->output);
    for (int i = 0; i < neuron->weight_count; i++) {
        printf("  w[%d]=%f, b[%d]=%f\n", i, neuron->weights[i].slope, i, neuron->weights[i].bias);
    }
}

void print_layer(const Layer* layer) {
    printf("Layer: neurons=%d\n", layer->neuron_count);
    for (int i = 0; i < layer->neuron_count; i++) {
        print_neuron(&layer->neurons[i]);
    }
}

void print_neural_network(const NeuralNetwork* nn) {
    printf("Neural Network: layers=%d\n", nn->layer_count);
    for (int l = 0; l < nn->layer_count; l++) {
        printf(" Layer %d:\n", l);
        print_layer(&nn->layers[l]);
    }
}
