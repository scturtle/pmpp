#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__constant__ float c0 = 0.5f;
__constant__ float c1 = 0.1f;
__constant__ float c2 = 0.1f;
__constant__ float c3 = 0.1f;
__constant__ float c4 = 0.1f;
__constant__ float c5 = 0.05f;
__constant__ float c6 = 0.05f;

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
  unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i*N*N + j*N + k] = c0*in[i*N*N + j*N + k]
                         + c1*in[i*N*N + j*N + (k - 1)]
                         + c2*in[i*N*N + j*N + (k + 1)]
                         + c3*in[i*N*N + (j - 1)*N + k]
                         + c4*in[i*N*N + (j + 1)*N + k]
                         + c5*in[(i - 1)*N*N + j*N + k]
                         + c6*in[(i + 1)*N*N + j*N + k];
  }
}

void stencil_cpu(float* in, float* out, unsigned int N) {
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      for (unsigned int k = 0; k < N; ++k) {
        size_t index = i * N * N + j * N + k;
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
           out[index] = c0*in[index]
                      + c1*in[i*N*N + j*N + (k - 1)]
                      + c2*in[i*N*N + j*N + (k + 1)]
                      + c3*in[i*N*N + (j - 1)*N + k]
                      + c4*in[i*N*N + (j + 1)*N + k]
                      + c5*in[(i - 1)*N*N + j*N + k]
                      + c6*in[(i + 1)*N*N + j*N + k];
        } else {
          out[index] = 0.0f;
        }
      }
    }
  }
}

int main() {
  unsigned int N = 64;
  size_t dataSize = (size_t)N * N * N * sizeof(float);

  float *h_in, *h_out_gpu, *h_out_cpu;
  float *d_in, *d_out;

  h_in = (float*)malloc(dataSize);
  h_out_gpu = (float*)malloc(dataSize);
  h_out_cpu = (float*)malloc(dataSize);

  for (size_t i = 0; i < (size_t)N * N * N; ++i) {
    h_in[i] = (float)(rand() % 100) / 10.0f;
  }

  cudaMalloc((void**)&d_in, dataSize);
  cudaMalloc((void**)&d_out, dataSize);

  cudaMemcpy(d_in, h_in, dataSize, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, dataSize);

  dim3 threadsPerBlock(8, 8, 8);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                     (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

  stencil_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out_gpu, d_out, dataSize, cudaMemcpyDeviceToHost);

  stencil_cpu(h_in, h_out_cpu, N);

  bool mismatch = false;
  float epsilon = 1e-5f;
  for (size_t i = 0; i < (size_t)N * N * N; ++i) {
    if (fabsf(h_out_gpu[i] - h_out_cpu[i]) > epsilon) {
      mismatch = true;
      printf("Mismatch found at index %zu: GPU=%.6f, CPU=%.6f\n", i, h_out_gpu[i], h_out_cpu[i]);
      break;
    }
  }

  if (mismatch) {
    printf("Verification FAILED\n");
  } else {
    printf("Verification PASSED\n");
  }

  cudaFree(d_in);
  cudaFree(d_out);
  free(h_in);
  free(h_out_gpu);
  free(h_out_cpu);

  return 0;
}
