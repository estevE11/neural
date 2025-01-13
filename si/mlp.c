#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// bg model
// l1 -> Linear(3, 5)
// act1 -> ReLu
// l2 -> Linear(5, 2)
// act2 -> ReLu
// l3 -> Linear(2, 2)
// act3 -> Sigmoid
//


// Structure for a layer
typedef struct {
    int num_inputs;     // Number of inputs to the layer
    int num_neurons;    // Number of neurons in the layer
    double *weights;    // 1D array storing weights (flattened matrix)
    double *biases;     // 1D array storing biases
    double *outputs;    // 1D array storing outputs of this layer
} Layer;

// Structure for the network
typedef struct {
    int num_layers;     // Number of layers
    Layer *layers;      // Array of layers
} NeuralNetwork;

double relu(double x) {
    return x > 0 ? x : 0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void softmax(double *x, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += exp(x[i]);
    }
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i]) / sum;
    }
}

void initLayer(Layer *layer, int n_conn, int n_neurons) {
    layer->num_inputs = n_conn;
    layer->num_neurons = n_neurons;
    layer->weights = (double *)malloc(n_conn * n_neurons * sizeof(double));
    layer->biases = (double *)malloc(n_neurons * sizeof(double));
    layer->outputs = (double *)malloc(n_neurons * sizeof(double));

    // Initialize weights and biases with random values (e.g., between -1 and 1)
    for (int i = 0; i < n_conn * n_neurons; i++) {
        layer->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    for (int i = 0; i < n_neurons; i++) {
        layer->biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

double getWeight(Layer *layer, int i_conn, int i_neuron) {
    return layer->weights[i_neuron * layer->num_inputs + i_conn];
}

void initNN(NeuralNetwork *nn, int* layers, int n_layers) {
    nn->num_layers = n_layers-1;
    nn->layers = (Layer *)malloc(nn->num_layers * sizeof(Layer));

    for(int i = 0; i < nn->num_layers; i++) {
        initLayer(&(nn->layers[i]), layers[i], layers[i+1]); 
    }
}

void free_layer(Layer *layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->outputs);
}

void free_network(NeuralNetwork *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        free_layer(&nn->layers[i]);
    }
    free(nn->layers);
}

void forwardLayer(Layer *layer, double *inputs, int sig) {
    for(int i_neuron = 0; i_neuron < layer->num_neurons; i_neuron++) {
        double sum = 0;
        for(int i_conn = 0; i_conn < layer->num_inputs; i_conn++) {
            sum += inputs[i_conn] * getWeight(layer, i_conn, i_neuron);
        }
        layer->outputs[i_neuron] = sum + layer->biases[i_neuron];
        if(sig == 1) {
            layer->outputs[i_neuron] = sigmoid(layer->outputs[i_neuron]);
        } else {
            layer->outputs[i_neuron] = relu(layer->outputs[i_neuron]);
        }
        softmax(layer->outputs, layer->num_neurons);
    }
}

void forward(NeuralNetwork *nn, double *inputs, double **out) {
    forwardLayer(&(nn->layers[0]), inputs, 0);
    for(int i = 1; i < nn->num_layers; i++) {
        if(i == nn->num_layers-1) {
            forwardLayer(&(nn->layers[i]), nn->layers[i-1].outputs, 1);
            break;
        }
        forwardLayer(&(nn->layers[i]), nn->layers[i-1].outputs, 0);
    }

    int out_len = nn->layers[nn->num_layers-1].num_neurons;
    *out = (double *)malloc(out_len * sizeof(double));
    for(int oi = 0; oi < out_len; oi++) {
        (*out)[oi] = nn->layers[nn->num_layers-1].outputs[oi];
    }
}

int main() {
    int layers[] = {3, 5, 2};

    NeuralNetwork nn;

    initNN(&nn, layers, 3);

    double *inputs = (double *)malloc(3 * sizeof(double));
    inputs[0] = 0.4;
    inputs[1] = 0.1;
    inputs[2] = 0.9;

    double *out;

    forward(&nn, inputs, &out);

    for(int i = 0; i < 2; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");

    free(out);
    free_network(&nn);

    return 0;
}
