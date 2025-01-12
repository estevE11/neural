#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Perceptron is based on y = ax + b

float w;
float b;

float targetFunction(float x) {
    return 3*x+10;
}

float forward(float x) {
    return w*x+b;
}

void train(float lr) {
    int train_data = 20;
    int steps = 1000;

    for(int i = 0; i < steps; i++) {
        int x = rand() % train_data;

        float X = forward(x);
        float Y = targetFunction(x);

        if(i % 100 == 0) {
            printf("step: %d\n", i);
            printf("X: %f, Y: %f\n", X, Y);
            printf("loss: %f\n\n", (Y-X));
        }

        w = w + lr * (Y - X) * ((float)x/(float)train_data);
        b = b + lr * (Y - X);
    }
}

int main() {
    srand(time(NULL));
    
    w = rand() % 10;
    b = rand() % 10;

    train(0.02);

    float X = forward(2);
    float Y = targetFunction(2);

    printf("loss: %f\n\n", (Y-X));

    return 0;
}


