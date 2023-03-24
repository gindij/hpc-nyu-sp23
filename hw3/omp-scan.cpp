#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void printarr(const long* x, long n) {
  for (long i = 0; i < n; i++) {
    printf("%d\n", x[i]);
  }
  printf("=========\n");
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  int p = 128;
  int block_size = n / p + 1;

  long* sums = (long*) malloc((p + 1) * sizeof(long));
  for (int i = 0; i < p + 1; i++) sums[i] = 0;

  double tt = omp_get_wtime();
  #pragma omp parallel num_threads(p)
  {
    #pragma omp for schedule(static, block_size) nowait
    for (long i = 0; i < n; i++) {
      long t = omp_get_thread_num();
      sums[t+1] += A[i];
    }

    #pragma omp for schedule(static, block_size)
    for (long i = 0; i < n; i++) {
      prefix_sum[i] = (i % block_size > 0) * (prefix_sum[i-1] + A[i-1]);
    }
  }
  printf("first par section time = %fs\n", omp_get_wtime() - tt);

  // printarr(prefix_sum, n);
  // printarr(sums, p);

  tt = omp_get_wtime();
  // accumulate the sums serially
  for (int t = 1; t < p; t++) {
    sums[t] += sums[t-1];
  }
  printf("accumulation time = %fs\n", omp_get_wtime() - tt);


  tt = omp_get_wtime();
  #pragma omp parallel num_threads(p)
  {
    int t = omp_get_thread_num();
    for (long di = 0; di < block_size; di++) {
      long i = t * block_size + di;
      if (i < n) prefix_sum[i] += sums[t];
    }
  }
  printf("first par section time = %fs\n", omp_get_wtime() - tt);


  free(sums);
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = i; rand();
  for (long i = 0; i < N; i++) B1[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}