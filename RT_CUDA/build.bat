nvcc -c Source.cu
nvcc -c kernel.cu
nvcc -o executable Source.obj kernel.obj