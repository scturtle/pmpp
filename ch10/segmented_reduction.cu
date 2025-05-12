#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 256

__global__ void SegmentedSumReductionKernel(float *input, float *output) {
  __shared__ float input_s[BLOCK_DIM];

  unsigned int segment = 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;

  input_s[t] = input[i] + input[i + BLOCK_DIM];

  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}

float cpu_sum_reduction(const float *input, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += input[i];
  }
  return sum;
}

int main() {
  const int NUM_BLOCKS = 128;
  const int N = 2 * BLOCK_DIM * NUM_BLOCKS;

  float *h_input = (float *)malloc(N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    h_input[i] = (float)i;
  }

  float *d_input;
  float *d_output_gpu;
  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_output_gpu, sizeof(float));

  float initial_output = 0.0f;
  cudaMemcpy(d_output_gpu, &initial_output, sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

  SegmentedSumReductionKernel<<<NUM_BLOCKS, BLOCK_DIM>>>(d_input, d_output_gpu);
  cudaDeviceSynchronize();

  float h_output_gpu = 0.0f;
  cudaMemcpy(&h_output_gpu, d_output_gpu, sizeof(float),
             cudaMemcpyDeviceToHost);

  float h_output_cpu = cpu_sum_reduction(h_input, N);

  float tolerance = 1e-5;
  if (std::fabs(h_output_gpu - h_output_cpu) < tolerance) {
    std::cout << "Results match!" << std::endl;
  } else {
    std::cout << "Results mismatch!" << std::endl;
  }

  free(h_input);
  cudaFree(d_input);
  cudaFree(d_output_gpu);
  return 0;
}
