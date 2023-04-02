CC = g++
OPTLEVEL = -O3
CFLAGS = -Wall

all: scan jac gs

scan:
	$(CC) $(CFLAGS) $(OPTLEVEL) -fopenmp omp-scan.cpp -o omp-scan

jac:
	$(CC) $(CFLAGS) $(OPTLEVEL) -fopenmp utils.cpp jacobi2D-omp.cpp -o jacobi2D-omp

gs:
	$(CC) $(CFLAGS) $(OPTLEVEL) -fopenmp utils.cpp gs2D-omp.cpp -o gs2D-omp

.PHONY: clean
clean:
	rm -f omp-scan jacobi2D-omp gs2D-omp