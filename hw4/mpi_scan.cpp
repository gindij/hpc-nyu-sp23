#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

// copied from hw3
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void printarr(const long* x, long n, int rank) {
  for (long i = 0; i < n; i++) {
    printf("rank %d: %ld\n", rank, x[i]);
  }
  printf("=========\n");
}

void scan_mpi(long* prefix_sum, const long* A, const long n) {

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  long chunk_size = n / world_size;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const long* A_send = NULL;
  if (rank == 0)
    A_send = A;

  long* partial_A = (long*) malloc(chunk_size * sizeof(long));

  MPI_Scatter(
    A_send,     // data to send
    chunk_size, // how much to send to each process
    MPI_LONG,   // type to send
    partial_A,  // buffer to receive partial data
    chunk_size, // how much each process receives
    MPI_LONG,   // type to receive
    0,          // sending process
    MPI_COMM_WORLD
  );

  // everyone: allocates and initializes partial sum array
  long* partial_prefix_sum = (long*) malloc(chunk_size * sizeof(long));
  for (long i = 0; i < chunk_size; i++) partial_prefix_sum[i] = 0;

  // everyone: run local scan
  scan_seq(partial_prefix_sum, partial_A, chunk_size);

  // everyone: compute individual offsets
  long offset = 0;
  for (long i = 0; i < chunk_size; i++) offset += partial_A[i];

  // everyone to everyone: communicate offsets
  long* offsets = (long*) malloc(world_size * sizeof(long));
  MPI_Allgather(&offset, 1, MPI_LONG, offsets, 1, MPI_LONG, MPI_COMM_WORLD);

  // everyone: add offsets to partial prefix sums
  for (long r = 0; r < rank; r++) {
    for (long i = 0; i < chunk_size; i++) {
      partial_prefix_sum[i] += offsets[r];
    }
  }

  // process 0: gather updated partial prefix sums
  MPI_Gather(partial_prefix_sum, chunk_size, MPI_LONG, prefix_sum, chunk_size, MPI_LONG, 0, MPI_COMM_WORLD);

  free(offsets);
  free(partial_A);
  free(partial_prefix_sum);

}

int main() {

  MPI_Init(NULL, NULL);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  long N = 64000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = i;
  for (long i = 0; i < N; i++) B1[i] = 0;

  if (rank == 0) {
    printf("N                 = %ld\n", N);
    printf("world size        = %d\n", world_size);
  }

  double tt;

  if (rank == 0) {
    tt = MPI_Wtime();
    scan_seq(B0, A, N);
    printf("sequential-scan   = %fs\n", MPI_Wtime() - tt);
  }

  tt = MPI_Wtime();
  scan_mpi(B1, A, N);
  if (rank == 0)
    printf("mpi parallel scan = %fs\n", MPI_Wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  if (rank == 0) printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);

  MPI_Finalize();

  return 0;
}
