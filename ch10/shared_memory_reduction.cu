#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

#define BLOCK_DIM 256
#define DATA_SIZE (BLOCK_DIM * 2)

__global__ void SharedMemorySumReductionKernel(float *input, float *output) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int t = threadIdx.x;

  input_s[t] = input[t] + input[t + BLOCK_DIM];
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  if (threadIdx.x == 0) {
    *output = input_s[0];
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
  std::vector<float> h_input(DATA_SIZE);

  for (int i = 0; i < DATA_SIZE; ++i) {
    h_input[i] = 1.0f;
  }

  float cpu_result = cpuSumReduction(h_input);

  float *d_input = nullptr;
  float *d_output = nullptr;
  float h_output = 0.0f;

  cudaMalloc(&d_input, DATA_SIZE * sizeof(float));
  cudaMalloc(&d_output, sizeof(float));

  cudaMemcpy(d_input, h_input.data(), DATA_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_output);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  bool passed = std::abs(cpu_result - h_output) < 1e-5;
  if (passed) {
    std::cout << "Comparison Passed" << std::endl;
  } else {
    std::cout << "Comparison Failed" << std::endl;
  }

  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}
