#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

__global__ void ConvergentSumReductionKernel(float *input, float *output) {
  unsigned int i = threadIdx.x;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

float cpuSumReduction(const std::vector<float> &input) {
  float sum = 0.0f;
  for (float val : input) {
    sum += val;
  }
  return sum;
}

int main() {
  const int N = 1024;
  const int blockSize = N / 2;
  const int gridSize = 1;

  std::vector<float> h_input(N);
  float h_cpu_output = 0.0f;
  float h_gpu_output = 0.0f;

  for (int i = 0; i < N; ++i) {
    h_input[i] = static_cast<float>(i + 1);
  }

  h_cpu_output = cpuSumReduction(h_input);

  float *d_input = nullptr;
  float *d_output = nullptr;

  cudaMalloc((void **)&d_input, N * sizeof(float));
  cudaMalloc((void **)&d_output, sizeof(float));

  cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
             cudaMemcpyHostToDevice);

  ConvergentSumReductionKernel<<<gridSize, blockSize>>>(d_input, d_output);

  cudaMemcpy(&h_gpu_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  const float epsilon = 1e-6f;
  if (std::fabs(h_cpu_output - h_gpu_output) < epsilon) {
    std::cout << "Results match!" << std::endl;
  } else {
    std::cout << "Results DO NOT match!" << std::endl;
  }

  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}
