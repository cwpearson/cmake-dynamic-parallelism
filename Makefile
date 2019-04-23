all: dp

NVCC_FLAGS = -arch=sm_35 -dc -std=c++11 -Xcompiler=-Wall -g 

%.o : %.cu
	nvcc ${NVCC_FLAGS} $< -o $@

dp: main.o dp.o
	nvcc -arch=sm_35 $^ -o $@

clean:
	rm -f main.o dp.o dp