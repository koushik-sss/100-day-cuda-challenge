#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

__global__ void Matrix2DAdd(const float *device_matrix_a,
                            const float *device_matrix_b,
                            float *device_matrix_c, const int R, const int C) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index < R && col_index < C) {
    device_matrix_c[row_index * C + col_index] =
        device_matrix_a[row_index * C + col_index] +
        device_matrix_b[row_index * C + col_index];
  }
}

void matrixAddGPU(const float *host_a, const float *host_b, float *host_c,
                  int R, int C) {

  size_t size = R * C * sizeof(float);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;

  cudaMalloc(&device_a, size);
  cudaMalloc(&device_b, size);
  cudaMalloc(&device_c, size);

  cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_c, host_c, size, cudaMemcpyHostToDevice); // will be redundant, need not keep this. retaining in code so that it is highlighted when I revisit this later!

  dim3 threadsPerBlock = {16,16};
  // dim3 blocksPerGrid = (R * C + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocksPerGrid = {((C + threadsPerBlock.x - 1) / threadsPerBlock.x), ((R + threadsPerBlock.y - 1) / threadsPerBlock.y)};

  Matrix2DAdd<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c,
                                                  R, C);

  cudaDeviceSynchronize();

  cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
}

int main() {

  const int N = 10000;
  const int R = N;
  const int C = N/2;
  float *host_a = new float[R * C];
  float *host_b = new float[R * C];
  float *host_c = new float[R * C];

  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      int index = (i * C) + j;
      host_a[index] = 1.0f;
      host_b[index] = 2.0f;
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  matrixAddGPU(host_a, host_b, host_c, R, C);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  bool correct = true;

  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      int index = (i * C) + j;
      if (host_c[index] != 3.0f) {
        correct = false;
        std::cout << "GPU 2D matrix addition error at index" << index << ":"
                  << host_c[index] << "!=3" << std::endl;
        break;
      }
    }
  }

  if (correct) {
    std::cout << "2d matrixadd completed successfully!" << std::endl;
  }

  delete[] host_a;
  delete[] host_b;
  delete[] host_c;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
