CXX = mpic++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

CPPTARGETS = $(basename $(wildcard *.cpp))

all: mpi_scan pingpong int_ring

%: %.cpp
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(CPPTARGETS) *.o

.PHONY: all, clean
