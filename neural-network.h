#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "activation-function.h"
// Multilayer Perceptron (MLP) Neural Network structures
// Case: Linear combination wX + b

// Estrutura para peso e bias
typedef struct {
    double slope; // peso para entrada x 
} Weight;

// Estrutura para neurônio
typedef struct {
    Weight* weights; 
    double bias;
    int weight_count; // normalmente igual ao número de entradas
    double (*activation_function)(double*, int); // função de ativação
    double output; // saída do neurônio
} Neuron;

// Estrutura para camada
typedef struct {
    Neuron* neurons;
    int neuron_count; // número de neurônios na camada
} Layer;

// Estrutura para rede neural
typedef struct {
    Layer* layers;
    int layer_count; // número de camadas
} NeuralNetwork;

// Funções de inicialização
void initialize_weights(Weight* weights, int count);
void initialize_neurons(Neuron* neurons, int neuron_count, int input_count, 
                        double (*activation_function)(double*, int));
void initialize_layer(Layer* layer, int neuron_count, int input_count, 
                      double (*activation_function)(double*, int));

void initialize_neural_network(NeuralNetwork* nn, int layer_count, int* neurons_per_layer, 
                               double (**activation_functions)(double*, int)); 
 
// Utilitários
double random_weight();

// Criação e liberação de estruturas
Neuron* create_neuron(int input_count, double (*activation_function)(double*, int));
void free_neuron(Neuron* neuron);

Layer* create_layer(int neuron_count, int input_count, 
                    double (**activation_functions)(double*, int));
void free_layer(Layer* layer);

NeuralNetwork* create_neural_network(int layer_count, int* neurons_per_layer, 
                                     double (**activation_functions)(double*, int));
void free_neural_network(NeuralNetwork* nn);

// Forward propagation
double forward_propagation(Layer* prev_layer, Layer* next_layer, const double* input);


// Backward propagation
void backward_propagation(NeuralNetwork* nn, const double* expected_output, double learning_rate);


// Debug
void print_neural_network(const NeuralNetwork* nn);
void print_layer(const Layer* layer);
void print_neuron(const Neuron* neuron);

#endif // NEURAL_NETWORK_H