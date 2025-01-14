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
    double *weights;    // 1D array storing weights (flattened matrix shape: num_inputs x num_neurons)
    double *biases;     // 1D array storing biases (shape: 1 x num_neurons)
    double *outputs;    // 1D array storing outputs of this layer (shape: 1 x num_neurons)
} Layer;

// Structure for the network
typedef struct {
    int num_layers;     // Number of layers
    Layer *layers;      // Array of layers
} NeuralNetwork;

void printArray(double *arr, int n) {
    printf("[");
    for (int i = 0; i < n-1; i++) {
        printf("%f, ", arr[i]);
    }
    printf("%f]", arr[n-1]);
    printf("\n");
}

void printMatrix(double *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        printArray(mat + i * cols, cols);
    }
}

void matmul(double *a, double *b, double *c, int ah, int aw, int bw) { // result matrix is: aw X bh
    for (int i = 0; i < ah; i++) {
        for (int j = 0; j < bw; j++) {
            double sum = 0;
            for (int k = 0; k < aw; k++) {
                sum += a[i * ah + k] * b[k * bw + j];
            }
            c[i * bw + j] = sum;
        }
    }
}

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
    matmul(inputs, layer->weights, layer->outputs, 1, layer->num_inputs, layer->num_neurons);

    // Biases and activation function
    for (int i = 0; i < layer->num_neurons; i++) {
        layer->outputs[i] += layer->biases[i];
        if(sig == 1) {
            layer->outputs[i] = sigmoid(layer->outputs[i]);
        } else {
            layer->outputs[i] = relu(layer->outputs[i]);
        }
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

    Layer *last_layer = &(nn->layers[nn->num_layers-1]);
    softmax(last_layer->outputs, last_layer->num_neurons);

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

    printf("Inputs: ");
    printArray(inputs, 3);

    double *out;

    forward(&nn, inputs, &out);

    printf("Outputs: ");
    printArray(out, 2);

    free(out);
    free_network(&nn);

    return 0;
}
