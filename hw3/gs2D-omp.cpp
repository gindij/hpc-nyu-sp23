#include <algorithm>
#include <stdio.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utils.h"

#ifdef _OPENMP
void solve(double** u, long n, const long maxiter, const double eps) {

    long    total_iters      = 0;
    double  initial_res_norm = residual_norm(u, n);
    double  h                = 1.0 / (n + 1);
    int     p                = 10;

    for (long iter = 0; iter < maxiter; iter++) {
        #pragma omp parallel num_threads(p)
        {
            // red
            #pragma omp for nowait
            for (long i = 0; i < n * n; i+=2) {
                double s = 0.0;
                if (i >= n       ) s += (*u)[i-n];
                if (i < n * n - n) s += (*u)[i+n];
                if (i >= 1       ) s += (*u)[i-1];
                if (i < n * n - 1) s += (*u)[i+1];
                (*u)[i] = 0.25 * (h * h + s);
            }

            // black
            #pragma omp for
            for (long i = 1; i < n * n; i+=2) {
                double s = 0.0;
                if (i >= n       ) s += (*u)[i-n];
                if (i < n * n - n) s += (*u)[i+n];
                if (i >= 1       ) s += (*u)[i-1];
                if (i < n * n - 1) s += (*u)[i+1];
                (*u)[i] = 0.25 * (h * h + s);
            }
        }
        double res_norm = residual_norm(u, n);
        // printf("res norm: %f\n", res_norm);
        if (res_norm / initial_res_norm < eps) break;
        total_iters = iter;
    }

    printf("total iters: %ld\n", total_iters + 1);
}
#else
void solve(double** u, long n, const long maxiter, const double eps) {

    long    total_iters      = 0;
    double  initial_res_norm = residual_norm(u, n);
    double  h                = 1.0 / (n + 1);

    for (long iter = 0; iter < maxiter; iter++) {

        for (long i = 0; i < n * n; i++) {
            double s = 0.0;
            if (i >= n       ) s += (*u)[i-n];
            if (i < n * n - n) s += (*u)[i+n];
            if (i >= 1       ) s += (*u)[i-1];
            if (i < n * n - 1) s += (*u)[i+1];
            (*u)[i] = 0.25 * (h * h + s);
        }

        double res_norm = residual_norm(u, n);
        // printf("res norm: %f\n", res_norm);
        if (res_norm / initial_res_norm < eps) break;
        total_iters = iter;
    }

    printf("total iters: %ld\n", total_iters + 1);
}
#endif

int main() {
    long         Ns[]    = {10, 20, 40, 80, 160, 320, 640, 1280, 2560};
    int          n       = 8;
    const long   maxiter = 5000;
    const double eps     = 1e-4;

    // solve(&u, N, maxiter, eps);

    // if (N <= 10) {
    //     for (long i = 0; i < N; i++) {
    //         for (long j = 0; j < N; j++) {
    //             printf("%f ", u[i*N+j]);
    //         }
    //         printf("\n");
    //     }
    // }

    for (int i = 0; i < n; i++) {

        long N = Ns[i];
        double* u = (double*) malloc(N * N * sizeof(double));
        for (long j = 0; j < N * N; j++) u[j] = 0.0;

        double tt = omp_get_wtime();
        solve(&u, N, maxiter, eps);
        free(u);
        printf("N = %ld: %fs\n", N, omp_get_wtime() - tt);

    }

    return 0;
}