#include <iostream>

__global__ void doubleElements(int *array, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        array[index] = array[index] * 2;
        
    }
}

int main() {
    const int arraySize = 15;
    int hostArray[arraySize] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
    int* deviceArray = nullptr;
    std::cout<<"device array before cudaMalloc is: "<<deviceArray<<std::endl;
    

    cudaMalloc(&deviceArray, arraySize*sizeof(int));
    std::cout<<"device array after cudaMalloc is: "<<deviceArray<<std::endl;


    cudaMemcpy(deviceArray, hostArray, arraySize*sizeof(int), cudaMemcpyHostToDevice);

    std::cout <<" Launching kernel " << std::endl;

    doubleElements<<<3,5>>>(deviceArray, arraySize);

    cudaDeviceSynchronize();


    cudaMemcpy(hostArray, deviceArray, arraySize*sizeof(int), cudaMemcpyDeviceToHost);

    
    std::cout << "Doubled array: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << hostArray[i] << " ";
    }
    std::cout << std::endl;
    
    
    cudaFree(deviceArray);
    
    return 0;

}