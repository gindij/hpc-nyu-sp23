CC = g++
OPTLEVEL = -O3
CFLAGS = -Wall
OMP_CHECK = $(shell echo | $(CC) -fopenmp -E - >/dev/null 2>&1 && echo "true" || echo "false")

ifeq ($(OMP_CHECK), true)
  OMP_FLAGS = -fopenmp
else
  OMP_FLAGS =
  $(warning OpenMP is not supported by the compiler, building without OpenMP support)
endif

all: jac gs scan

scan:
	$(CC) $(CFLAGS) $(OPTLEVEL) $(OMP_FLAGS) omp-scan.cpp -o omp-scan
        
jac:
	$(CC) $(CFLAGS) $(OPTLEVEL) $(OMP_FLAGS) utils.cpp jacobi2D-omp.cpp -o jacobi2D-omp

gs:
	$(CC) $(CFLAGS) $(OPTLEVEL) $(OMP_FLAGS) utils.cpp gs2D-omp.cpp -o gs2D-omp

.PHONY: clean
clean:
	rm -f omp-scan jacobi2D-omp gs2D-omp
