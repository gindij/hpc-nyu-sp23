#include <math.h>
#include "utils.h"

#ifndef UTILS_H
#define UTILS_H

double residual_norm(double** u, long n) {
    double h = 1.0 / (n + 1);
    double sq_norm = 0.0;
    for (long i = 0; i < n * n; i++) {
        double Au = 4 * (*u)[i];
        if (i > n - 1    ) Au += -(*u)[i-n];
        if (i < n * n - n) Au += -(*u)[i+n];
        if (i > 0        ) Au += -(*u)[i-1];
        if (i < n * n - 1) Au += -(*u)[i+1];
        double norm_term = Au / (h * h) - 1;
        double sq_term = norm_term * norm_term;
        sq_norm += sq_term;
    }
    return sqrt(sq_norm);
}

#endif