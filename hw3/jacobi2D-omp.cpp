#include <algorithm>
#include <stdio.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utils.h"

void solve(double** u, long n, const long maxiter, const double eps) {

    long    total_iters      = 0;
    double  initial_res_norm = residual_norm(u, n);
    double* u_next           = (double*) malloc(n * n * sizeof(double));
    double  h                = 1.0 / (n + 1);

    for (long i = 0; i < n * n; i++) u_next[i] = 0.0;

    for (long iter = 0; iter < maxiter; iter++) {

        #pragma omp parallel
        {
            #pragma omp for
            for (long i = 0; i < n * n; i++) {
                double s = 0.0;
                if (i >= n       ) s += (*u)[i-n];
                if (i < n * n - n) s += (*u)[i+n];
                if (i >= 1       ) s += (*u)[i-1];
                if (i < n * n - 1) s += (*u)[i+1];
                u_next[i] = 0.25 * (h * h + s);
            }

            #pragma omp for
            for (long i = 0; i < n * n; i++) (*u)[i] = u_next[i];
        }
        double res_norm = residual_norm(&u_next, n);
        // printf("res norm: %f\n", res_norm);
        if (res_norm / initial_res_norm < eps) break;
        total_iters = iter;
    }

    free(u_next);

    printf("total iters: %ld\n", total_iters + 1);
}

int main() {
    long         N       = 1000;
    const long   maxiter = 5000;
    const double eps     = 1e-4;
    double*      u       = (double*) malloc(N * N * sizeof(double));

    for (long i = 0; i < N * N; i++) u[i] = 0.0;

    solve(&u, N, maxiter, eps);

    if (N <= 10) {
        for (long i = 0; i < N; i++) {
            for (long j = 0; j < N; j++) {
                printf("%f ", u[i*N+j]);
            }
            printf("\n");
        }
    }

    free(u);

    return 0;
}