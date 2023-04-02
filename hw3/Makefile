CC = g++
OPTLEVEL = -O3
CFLAGS = -Wall

ifeq ($(shell grep -c _OPENMP $(CC) 2> /dev/null), 1)
    CFLAGS += -fopenmp
endif

scan:
	$(CC) $(CFLAGS) $(OPTLEVEL) -fopenmp omp-scan.cpp -o omp-scan

jac:
	$(CC) $(CFLAGS) $(OPTLEVEL) -fopenmp utils.cpp jacobi2D-omp.cpp -o jacobi2D-omp

gs:
	$(CC) $(CFLAGS) $(OPTLEVEL) -fopenmp utils.cpp gs2D-omp.cpp -o gs2D-omp

.PHONY: clean
clean:
	rm -f omp-scan jacobi2D-omp gs2D-omp