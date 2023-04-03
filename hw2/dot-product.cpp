#include <math.h>
#include <stdio.h>
#include "utils.h"
#include <omp.h>

#define BLOCK_SIZE 16

double naive_dot(double* v1, double* v2, long N) {
    double s;
    for (long i = 0; i < N; i++) {
        s += v1[i] * v2[i];
    }
    return s;
}

double pipelined_dot(double* u, double* v, long N) {

    double o[BLOCK_SIZE];
    double uu[BLOCK_SIZE];
    double vv[BLOCK_SIZE];
    double s[BLOCK_SIZE];

    // fetch + multiply
    #pragma omp unroll BLOCK_SIZE
    for (int i = 0; i < BLOCK_SIZE; i++) {
        o[i] = u[i] * v[i];
        uu[i] = u[i+BLOCK_SIZE];
        vv[i] = v[i+BLOCK_SIZE];
    }

    for (long i = 0; i < N; i+=BLOCK_SIZE) {
        #pragma omp unroll BLOCK_SIZE
        // add, multiply, fetch
        for (int j = 0; j < BLOCK_SIZE; j++) {
            s[j] = s[j] + o[j];
            o[j] = uu[j] * vv[j];
            uu[j] = u[i+BLOCK_SIZE+j];
            vv[j] = v[i+BLOCK_SIZE+j];
        }
    }

    // add + multiply
    #pragma omp unroll BLOCK_SIZE
    for (int i = 1; i < BLOCK_SIZE; i++) {
        s[i] = s[i-1] + s[i] + o[i];
    }

    return s[BLOCK_SIZE-1];
}


void profile_dot_method(long n, int repeats, double (*test_dot)(double*, double*, long)) {

    double* v1 = (double*) malloc(n * sizeof(double));
    double* v2 = (double*) malloc(n * sizeof(double));
    double error = 0;

    Timer t;

    double total_time = 0.0;
    for (int i = 0; i < repeats; i++) {
        for (long j = 0; j < n; j++) {
            v1[j] = drand48();
            v2[j] = drand48();
        }
        double naive_result = naive_dot(v1, v2, n);

        t.tic();
        double test_result = test_dot(v1, v2, n);
        total_time += t.toc();

        error += abs(naive_result - test_result);
    }

    printf("time per dot: %.5fs\n", total_time / repeats);
    printf("avg error: %f\n", error / repeats);

    free(v1);
    free(v2);
}


int main(int argc, char** argv) {

    long NREPEATS = read_option<long>("-nrepeats", argc, argv);
    long N = read_option<long>("-N", argc, argv);

    profile_dot_method(N, NREPEATS, &naive_dot);
    profile_dot_method(N, NREPEATS, &pipelined_dot);

    return 0;
}