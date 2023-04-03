#include <algorithm>
#include <stdio.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utils.h"

#ifdef _OPENMP
long solve(double** u, long n, const long maxiter, const double eps, const int p) {

    long    total_iters      = 0;
    double  initial_res_norm = residual_norm(u, n);
    double  h                = 1.0 / (n + 1);

    for (long iter = 0; iter < maxiter; iter++) {
        #pragma omp parallel num_threads(p)
        {
            // red
            #pragma omp for
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

    return total_iters;
}
#else
long solve(double** u, long n, const long maxiter, const double eps, const int p) {

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

    return total_iters;
}
#endif

int main() {
    long         Ns[]    = {10, 20, 40, 80, 160, 320, 640, 1280};
    int          ps[]    = {1, 2, 4, 6, 8, 16, 32};
    int          np      = 7;
    int          nN      = 8;
    const long   maxiter = 5000;
    const double eps     = 1e-4;

    // double* u = (double*) malloc(3 * 3 * sizeof(double));
    // solve(&u, 3L, 5000, eps);
    // for (long i = 0; i < 3L; i++) {
    //     for (long j = 0; j < 3L; j++) {
    //         printf("%f ", u[i*3+j]);
    //     }
    //     printf("\n");
    // }
    // free(u);
    for (int i = 0; i < nN; i++) {
        long N = Ns[i];
        printf("N = %ld\t", N);
        for (int j = 0; j < np; j++) {
            int p = ps[j];
            double* u = (double*) malloc(N * N * sizeof(double));
            for (long j = 0; j < N * N; j++) u[j] = 0.0;

            double tt = omp_get_wtime();
            solve(&u, N, maxiter, eps, p);
            free(u);
            printf("p = %d: %fs  ", p, omp_get_wtime() - tt);
        }
        printf("\n");
    }

    return 0;
}