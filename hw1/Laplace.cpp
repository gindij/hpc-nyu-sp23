#include <stdio.h>
#include <math.h>
#include "utils.h"

double res_norm(double* u, int n) {
    double h2 = 1. / ((n + 1) * (n + 1));
    double res = 0;
    double Aui;

    for (int i = 0; i < n; i++) {
        Aui = 2 * u[i];
        if (i - 1 >= 0) Aui -= u[i-1];
        if (i + 1 <  n) Aui -= u[i+1];
        Aui /= h2;
        res += (Aui - 1) * (Aui - 1);
    }

    return sqrt(res);
}

void jacobi_step(double **u, double** u_next, int n) {

    double h2 = 1. / ((n + 1) * (n + 1));
    double s;

    for (int i = 0; i < n; i++) {
        s = 0;
        if (i - 1 >= 0) s += -(*u)[i-1];
        if (i + 1 < n)  s += -(*u)[i+1];
        (*u_next)[i] = (h2 - s) / 2;
    }

    for (int i = 0; i < n; i++) {
        (*u)[i] = (*u_next)[i];
    }
}

void gauss_seidel_step(double** u, double** u_next, int n) {
    
    double h2 = 1. / ((n + 1) * (n + 1));
    double s;

    for (int i = 0; i < n; i++) {
        s = 0;
        if (i - 1 >= 0) s += -(*u)[i-1];
        if (i + 1 <  n) s += -(*u)[i+1];
        (*u)[i] = (h2 - s) / 2;
    }
}

int solve(int n, int max_iter, double epsilon, void (*step_fn)(double**, double**, int), int verbose) {

    double* u      = (double*) calloc(n, sizeof(double));
    double* u_next = (double*) calloc(n, sizeof(double));

    int iter;
    double init_res_norm = res_norm(u, n);
    for (iter = 0; iter < max_iter; iter++) {
        step_fn(&u, &u_next, n);
        double r_norm = res_norm(u, n);
        if (verbose)
            printf(
                "iteration %d: %5f (decrease factor = %5f)\n", 
                iter + 1, 
                r_norm, 
                init_res_norm / r_norm
            );
        if (r_norm / init_res_norm < epsilon) 
            break;
        iter += 1;
    }

    free(u);
    free(u_next);

    return iter;
}


int main(int argc, char** argv) {

    const int    N        = read_option<int>("-N", argc, argv);
    const int    MAX_ITER = read_option<int>("-maxiter", argc, argv);
    const int    VERBOSE  = read_option<int>("-v", argc, argv);
    const double EPSILON  = 1e-4;
    
    Timer t2;
    t2.tic();
    int iters_gs = solve(N, MAX_ITER, EPSILON, &gauss_seidel_step, VERBOSE);
    double time_gs = t2.toc();
    printf("Gauss-Seidel (N = %d, max iters = %d): time = %f, iters = %d\n", N, MAX_ITER, time_gs, iters_gs);

    return 0;
}