#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define BLOCK_SIZE 32  // Adjust this based on your CPU's cache size

void transposed(double *a, double *c, int w, int h);
void matmul(double *a, double *b, double *c, int aw, int ah, int bw);
void mem_opt_matmul(double *a, double *b, double *c, int aw, int ah, int bw);
void block_matmul(double *a, double *b, double *c, int aw, int ah, int bw);
void print_matrix(double *a, int w, int h);
int matrices_equal(double *a, double *b, int size, double epsilon);

void matmul(double *a, double *b, double *c, int aw, int ah, int bw) { // result matrix is: aw X bh
    for (int i = 0; i < ah; i++) {
        for (int j = 0; j < bw; j++) {
            double sum = 0;
            for (int k = 0; k < aw; k++) {
                int a_index = i * ah + k;
                int b_index = k * bw + j;
                //printf("loading A: %d, B: %d\n", a_index, b_index);
                sum += a[a_index] * b[b_index];
            }
            c[i * bw + j] = sum;
        }
    }
}

void mem_opt_matmul(double *a, double *b, double *c, int aw, int ah, int bw) {
    double *bt = malloc(sizeof(double) * bw * ah);
    transposed(b, bt, bw, ah);
    for (int i = 0; i < ah; i++) {
        for (int j = 0; j < bw; j++) {
            double sum = 0;
            for (int k = 0; k < aw; k++) {
                int a_index = i * ah + k;
                int b_index = j * bw + k;
                sum += a[a_index] * bt[b_index];
            }
            c[i * bw + j] = sum;
        }
    }
    free(bt);
}

#define BLOCK_SIZE 32
void block_matmul(double *a, double *b, double *c, int aw, int ah, int bw) {
    for (int i = 0; i < ah; i += BLOCK_SIZE) {
        for (int j = 0; j < bw; j += BLOCK_SIZE) {
            for (int k = 0; k < aw; k += BLOCK_SIZE) {
                // Compute block (i,j) of C
                for (int ii = i; ii < i + BLOCK_SIZE && ii < ah; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < bw; jj++) {
                        double sum = 0;
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < aw; kk++) {
                            sum += a[ii * aw + kk] * b[kk * bw + jj];
                        }
                        c[ii * bw + jj] += sum;
                    }
                }
            }
        }
    }
}

void transposed(double *a, double *c, int w, int h) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            c[j * h + i] = a[i * w + j];
        }
    }
}

void print_matrix(double *a, int w, int h) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f ", a[i * h + j]);
        }
        printf("\n");
    }
}

void generate_random_matrix(double *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

int matrices_equal(double *a, double *b, int size, double epsilon) {
    for (int i = 0; i < size * size; i++) {
        if (fabs(a[i] - b[i]) > epsilon) {
            printf("Matrices differ at index %d: %f != %f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    srand(time(NULL));  // Initialize random seed
    
    int size = 1024;
    double *a = malloc(sizeof(double) * size * size);
    double *b = malloc(sizeof(double) * size * size);
    double *c = malloc(sizeof(double) * size * size);
    double *c2 = malloc(sizeof(double) * size * size);
    double *c3 = malloc(sizeof(double) * size * size);

    generate_random_matrix(a, size);
    generate_random_matrix(b, size);

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    matmul(a, b, c, size, size, size);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Standard matmul took %f seconds\n", cpu_time_used);

    start = clock();
    mem_opt_matmul(a, b, c2, size, size, size);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Memory optimized matmul took %f seconds\n", cpu_time_used);

    start = clock();
    block_matmul(a, b, c3, size, size, size);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Block optimized matmul took %f seconds\n", cpu_time_used);

    // Verify results
    double epsilon = 1e-10;  // tolerance for floating point comparison
    if (!matrices_equal(c, c2, size, epsilon)) {
        printf("Memory optimized result differs from standard result!\n");
    } else {
        printf("Memory optimized result verified correct\n");
    }

    if (!matrices_equal(c, c3, size, epsilon)) {
        printf("Block optimized result differs from standard result!\n");
    } else {
        printf("Block optimized result verified correct\n");
    }

    // Free allocated memory
    free(a);
    free(b);
    free(c);
    free(c2);
    free(c3);
    
    return 0;
}
