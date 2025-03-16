#include <chrono>
#include <iostream>
#include <vector>

void VectorAddCPU(const float *a, const float *b, float *c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int N = 1000000;

  float *a = new float[N];
  float *b = new float[N];
  float *c = new float[N];

  for (auto i = 0; i < N; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();

  VectorAddCPU(a, b, c, N);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);


  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (c[i] != 3.0f) {
      correct = false;
      std::cout << "Error at index " << i << ": " << c[i] << " != 3.0"
                << std::endl;
      break;
    }
  }
  if (correct) {
    std::cout << "CPU vector addition completed successfully!" << std::endl;
  }

  std::cout << "CPU Time taken: " << duration.count() / 1000.0
            << " milliseconds" << std::endl;

  // Clean up
  delete[] a;
  delete[] b;
  delete[] c;

  return 0;
}
