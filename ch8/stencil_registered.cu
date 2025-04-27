#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define OUT_TILE_DIM 16
#define IN_TILE_DIM (OUT_TILE_DIM + 2)

const float c0 = 0.5f;
const float c1 = 0.05f;
const float c2 = 0.05f;
const float c3 = 0.1f;
const float c4 = 0.1f;
const float c5 = 0.1f;
const float c6 = 0.1f;

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
  int iStart = blockIdx.z*OUT_TILE_DIM;
  int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
  int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;

  float inPrev;
  __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
  float inCurr;
  float inNext;

  if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
    inPrev = in[(iStart - 1)*N*N + j*N + k];
  }

  if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
    inCurr = in[iStart*N*N + j*N + k];
    inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
  }

  for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
    if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
      inNext = in[(i + 1)*N*N + j*N + k];
    }
    __syncthreads();

    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
         threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
        out[i*N*N + j*N + k] = c0*inCurr_s[threadIdx.y][threadIdx.x]
                             + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                             + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                             + c3*inCurr_s[threadIdx.y+1][threadIdx.x]
                             + c4*inCurr_s[threadIdx.y-1][threadIdx.x]
                             + c5*inPrev
                             + c6*inNext;
      }
    }
    __syncthreads();

    inPrev = inCurr;
    inCurr = inNext;
    inCurr_s[threadIdx.y][threadIdx.x] = inNext;
  }
}

void stencil_cpu(const float* in, float* out, unsigned int N) {
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      for (unsigned int k = 0; k < N; ++k) {
        out[i * N * N + j * N + k] = 0.0f;
      }
    }
  }
  for (unsigned int i = 1; i < N - 1; ++i) {
    for (unsigned int j = 1; j < N - 1; ++j) {
      for (unsigned int k = 1; k < N - 1; ++k) {
        out[i * N * N + j * N + k] =
          c0 * in[i * N * N + j * N + k] +
          c1 * in[i * N * N + j * N + (k - 1)] +
          c2 * in[i * N * N + j * N + (k + 1)] +
          c3 * in[i * N * N + (j + 1) * N + k] +
          c4 * in[i * N * N + (j - 1) * N + k] +
          c5 * in[(i - 1) * N * N + j * N + k] +
          c6 * in[(i + 1) * N * N + j * N + k];
      }
    }
  }
}

int main() {
  const int N = 64;
  unsigned int dataSize = N * N * N;
  unsigned int memSize = dataSize * sizeof(float);

  float* h_in = (float*)malloc(memSize);
  float* h_out_gpu = (float*)malloc(memSize);
  float* h_out_cpu = (float*)malloc(memSize);

  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      for (unsigned int k = 0; k < N; ++k) {
        h_in[i * N * N + j * N + k] = (float)(rand() % 100) / 10.0f;
        h_out_gpu[i * N * N + j * N + k] = 0.0f;
      }
    }
  }

  float* d_in = NULL;
  float* d_out = NULL;

  cudaMalloc((void**)&d_in, memSize);
  cudaMalloc((void**)&d_out, memSize);

  cudaMemcpy(d_in, h_in, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out_gpu, memSize, cudaMemcpyHostToDevice);

  dim3 blockSize(IN_TILE_DIM, IN_TILE_DIM, 1);
  dim3 gridSize( (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM );

  stencil_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);

  cudaDeviceSynchronize();

  cudaMemcpy(h_out_gpu, d_out, memSize, cudaMemcpyDeviceToHost);

  stencil_cpu(h_in, h_out_cpu, N);

  bool ok = true;
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      for (unsigned int k = 0; k < N; ++k) {
        unsigned int index = i * N * N + j * N + k;
        float diff = h_out_gpu[index] - h_out_cpu[index];
        ok = ok && fabs(diff) <= 1e-5;
      }
    }
  }

  printf("Verification %s\n", ok ? "PASSED" : "FAILED");

  cudaFree(d_in);
  cudaFree(d_out);
  free(h_in);
  free(h_out_gpu);
  free(h_out_cpu);

  return 0;
}
