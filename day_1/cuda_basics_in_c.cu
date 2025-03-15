#include<stdio.h>

__global__ void helloFromGPU() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU! I am thread %d in block %d \n", threadIdx.x,  blockIdx.x);
}


int main() {

printf("Hello from CPU! \n");

int numBlocks = 2;
int threadsPerBlock = 4;

printf("Launching kernel with %d blocks, each with %d threads... \n", numBlocks, threadsPerBlock);
helloFromGPU <<<numBlocks, threadsPerBlock>>>();
cudaError_t error = cudaDeviceSynchronize();

if(error!=cudaSuccess) {
    printf("Cuda error: %s\n", cudaGetErrorString(error));
}

else {
    printf("execution was successful! \n");
}

cudaDeviceReset();

return 0;
}

