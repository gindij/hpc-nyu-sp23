#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

void ring_recieve_send_int(long* msg, int world_size, MPI_Comm comm) {

  int rank;
  MPI_Comm_rank(comm, &rank);

  MPI_Status status;

  // if the rank is > 0, receive a message from the previous process in the ring
  if (rank > 0) {
    MPI_Recv(msg, 1, MPI_DOUBLE, rank - 1, 1, comm, &status);
    // printf("rank = %d, received %d\n", rank, *msg);
  }

  // add the current rank to the message
  *msg += rank;

  // if send the new message on to the next process in the ring
  if (rank <= world_size - 1) {
    MPI_Send(msg, 1, MPI_DOUBLE, (rank + 1) % world_size, 1, comm);
    // printf("rank = %d, sent %d\n", rank, *msg);
  }

  // the rank 0 process should receive the final message
  if (rank == 0) {
    MPI_Recv(msg, 1, MPI_DOUBLE, world_size - 1, 1, comm, &status);
    // printf("rank = %d, received final message %d\n", rank, *msg);
  }

  return;
}

void ring_recieve_send_array(double** msg, long Nelts, int world_size, MPI_Comm comm) {

  int rank;
  MPI_Comm_rank(comm, &rank);

  MPI_Status status;

  // if the rank is > 0, receive a message from the previous process in the ring
  if (rank > 0) {
    MPI_Recv(*msg, Nelts, MPI_DOUBLE, rank - 1, 1, comm, &status);
    // printf("rank = %d, received %d\n", rank, *msg);
  }

  // add the current rank to the message
  for (long i = 0; i < Nelts; i++) (*msg)[i] += rank;

  // if send the new message on to the next process in the ring
  if (rank <= world_size - 1) {
    MPI_Send(*msg, Nelts, MPI_DOUBLE, (rank + 1) % world_size, 1, comm);
    // printf("rank = %d, sent %d\n", rank, *msg);
  }

  // the rank 0 process should receive the final message
  if (rank == 0) {
    MPI_Recv(*msg, Nelts, MPI_DOUBLE, world_size - 1, 1, comm, &status);
    // printf("rank = %d, received final message %d\n", rank, *msg);
  }

  return;
}

int main(int argc, char** argv) {

  if (argc < 3) {
    printf("Usage: mpirun ./int_ring send_array Nloops\n");
    abort();
  }
  int send_array = atoi(argv[1]);
  long Nloops = atol(argv[2]);
  long Nelts = 250000; // about 2mb

  double* msg_array = (double*) malloc(Nelts * sizeof(double));
  for (long i = 0; i < Nelts; i++) msg_array[i] = 0.12345;
  long msg_long = 0;

  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &world_size);

  double tt = MPI_Wtime();
  long loops_done = 0;
  while (loops_done < Nloops) {
    if (send_array)
      ring_recieve_send_array(&msg_array, Nelts, world_size, comm);
    else
      ring_recieve_send_int(&msg_long, world_size, comm);
    MPI_Barrier(comm);
    loops_done++;
    // printf("loops done = %d (out of %ld)\n", loops_done, Nloops);
  }
  double time_taken = MPI_Wtime() - tt;

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    printf("sending array = %d\n", send_array);
    printf("loops         = %ld\n", Nloops);
    printf("world size    = %d\n", world_size);
    printf("time          = %f\n", time_taken);
    if (send_array) {
      printf("elements      = %ld\n", Nelts);
      printf("result[0]     = %f\n", msg_array[0]);
    } else {
      printf("result        = %d\n", msg_long);
    }
  }

  MPI_Finalize();

  free(msg_array);
}