#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define CACHE_LINE_SIZE 64

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
    printf("%ld\n", x[i]);
  }
  printf("=========\n");
}

void scan_omp(long* prefix_sum, const long* A, long n, const int p) {
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  printf("num threads = %d\n", p);
  int block_size = n / p + 1;

  long* sums = (long*) malloc((p + 1) * sizeof(long));
  for (int i = 0; i < p + 1; i++) sums[i] = 0;

  #pragma omp parallel num_threads(p)
  {
    #pragma omp for schedule(static, block_size)
    for (long i = 0; i < n; i++) {
      long t = omp_get_thread_num();
      sums[t] += A[i];
    }
  }

  #pragma omp parallel num_threads(p)
  {
    long t = omp_get_thread_num();
    long start_idx = t * block_size;
    long end_idx = std::min((t + 1) * block_size, n);
    prefix_sum[start_idx] = 0;
    for (long i = start_idx + 1; i < end_idx; i++) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }

  // accumulate the sums serially
  for (int t = 1; t < p; t++) sums[t] += sums[t-1];

  #pragma omp parallel for schedule(static, block_size) num_threads(p)
  for (long i = block_size; i < n; i++) {
    int t = omp_get_thread_num();
    prefix_sum[i] += sums[t];
  }
  free(sums);
}

int main() {
  long N = 800000000;
  printf("N = %ld\n", N);
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan          = %fs\n", omp_get_wtime() - tt);

  for (int p = 1; p < omp_get_max_threads() * 2 + 1; p+=1) {
    for (long i = 0; i < N; i++) B1[i] = 0;
    tt = omp_get_wtime();
    scan_omp(B1, A, N, p);
    printf("parallel-scan (p = %d) = %fs\n", p, omp_get_wtime() - tt);
  }

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}