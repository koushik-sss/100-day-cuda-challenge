#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAddKernel(const float *a, const float *b, float *c,
                                int n) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void vectorAddGPU(const float *host_array_a, const float *host_array_b,
                  float *host_array_c, int n) {
  size_t size = n * sizeof(float);

  float *device_array_a = nullptr;
  float *device_array_b = nullptr;
  float *device_array_c = nullptr;

  cudaMalloc(&device_array_a, size);
  cudaMalloc(&device_array_b, size);
  cudaMalloc(&device_array_c, size);

  cudaMemcpy(device_array_a, host_array_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_array_b, host_array_b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_array_c, host_array_c, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 512;

  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(
      device_array_a, device_array_b, device_array_c, n);

  cudaDeviceSynchronize();

  cudaMemcpy(host_array_c, device_array_c, size, cudaMemcpyDeviceToHost);

  cudaFree(device_array_a);
  cudaFree(device_array_b);
  cudaFree(device_array_c);
}

int main() {
  const int N = 1000000;

  float *h_a = new float[N];
  float *h_b = new float[N];
  float *h_c = new float[N];

  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  vectorAddGPU(h_a, h_b, h_c, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);


  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (h_c[i] != 3.0f) {
      correct = false;
      std::cout << "Error at index " << i << ": " << h_c[i] << " != 3.0"
                << std::endl;
      break;
    }
  }

  if (correct) {
    std::cout << "GPU vector addition completed successfully!" << std::endl;
  }

  std::cout << "GPU Time taken: " << milliseconds << " milliseconds"
            << std::endl;


  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
