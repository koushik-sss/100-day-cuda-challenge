#include<iostream>

__global__ void helloFromGPU() {
    int threadId = blockIdx.x + blockDim.x + threadIdx.x;

    printf("Hello from GPU, thread %d!\n", threadId);
}

int main() {
    std::cout<<"Hello from the CPU" << std::endl;

    helloFromGPU<<<2,5>>>();

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    if(error!= cudaSuccess) {
        std::cerr <<"CUDA error" << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::cout << " Execution on GPU done!" << std::endl;

    return 0;
}