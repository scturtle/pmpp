#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

#define BLOCK_DIM 256
#define COARSE_FACTOR 4
#define N (BLOCK_DIM * COARSE_FACTOR * 2 * 64)

__global__ void CoarsenedSumReductionKernel(float *input, float *output) {
  __shared__ float input_s[BLOCK_DIM];

  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;

  float sum = input[i];
  for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
    sum += input[i + tile * BLOCK_DIM];
  }

  input_s[t] = sum;

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

int main() {
  std::vector<float> h_input(N);
  float h_output_cpu = 0.0f;
  float h_output_gpu = 0.0f;

  for (int i = 0; i < N; ++i) {
    h_input[i] = static_cast<float>(i % 100) + 0.5f;
  }

  for (int i = 0; i < N; ++i) {
    h_output_cpu += h_input[i];
  }

  float *d_input = nullptr;
  float *d_output_gpu = nullptr;

  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_output_gpu, sizeof(float));

  cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(d_output_gpu, 0, sizeof(float));

  dim3 blockDim(BLOCK_DIM);
  dim3 gridDim(N / (BLOCK_DIM * COARSE_FACTOR * 2));

  CoarsenedSumReductionKernel<<<gridDim, blockDim>>>(d_input, d_output_gpu);

  cudaMemcpy(&h_output_gpu, d_output_gpu, sizeof(float),
             cudaMemcpyDeviceToHost);

  if (std::abs(h_output_cpu - h_output_gpu) < 1e-5) {
    std::cout << "Results match." << std::endl;
  } else {
    std::cout << "Results do not match." << std::endl;
  }

  cudaFree(d_input);
  cudaFree(d_output_gpu);

  return 0;
}
