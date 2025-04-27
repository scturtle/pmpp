#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matrixMulKernel(float* M, float* N, float* P, int width)
{
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by*TILE_WIDTH + ty;
  int colStart = bx*TILE_WIDTH*COARSE_FACTOR + tx;

  float Pvalue[COARSE_FACTOR];
  for(int c = 0; c < COARSE_FACTOR; ++c) {
    Pvalue[c] = 0.0f;
  }

  for(int ph = 0; ph < width/TILE_WIDTH; ++ph) {
    Mds[ty][tx] = M[row*width + ph*TILE_WIDTH + tx];

    for(int c = 0; c < COARSE_FACTOR; ++c) {
      int col = colStart + c*TILE_WIDTH;
      Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*width + col];
      __syncthreads();

      for(int k = 0; k < TILE_WIDTH; ++k) {
        Pvalue[c] += Mds[ty][k]*Nds[k][tx];
      }
      __syncthreads();
    }
  }

  for(int c = 0; c < COARSE_FACTOR; ++c) {
    int col = colStart + c*TILE_WIDTH;
    P[row*width + col] = Pvalue[c];
  }
}

void matrixMulCPU(float* M, float* N, float* P, int width) {
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < width; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < width; ++k) {
        sum += M[i * width + k] * N[k * width + j];
      }
      P[i * width + j] = sum;
    }
  }
}

int main() {
  int width = 256;
  assert(width % TILE_WIDTH == 0);
  assert(width % (TILE_WIDTH * COARSE_FACTOR) == 0);

  int numElements = width * width;
  size_t size = numElements * sizeof(float);

  float *h_M, *h_N, *h_P_cpu, *h_P_gpu;
  float *d_M, *d_N, *d_P;

  h_M = (float*)malloc(size);
  h_N = (float*)malloc(size);
  h_P_cpu = (float*)malloc(size);
  h_P_gpu = (float*)malloc(size);

  for (int i = 0; i < numElements; ++i) {
    h_M[i] = (float)(rand() % 100) / 10.0f;
    h_N[i] = (float)(rand() % 100) / 10.0f;
  }

  cudaMalloc((void**)&d_M, size);
  cudaMalloc((void**)&d_N, size);
  cudaMalloc((void**)&d_P, size);

  cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim(width / (TILE_WIDTH * COARSE_FACTOR), width / TILE_WIDTH);

  matrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, width);

  cudaMemcpy(h_P_gpu, d_P, size, cudaMemcpyDeviceToHost);

  matrixMulCPU(h_M, h_N, h_P_cpu, width);

  double tolerance = 1e-3;
  int errors = 0;
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_P_cpu[i] - h_P_gpu[i]) > tolerance) {
      errors++;
    }
  }

  if (errors == 0) {
    printf("Verification Successful!\n");
  } else {
    printf("Verification FAILED! %d errors found.\n", errors);
  }

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
  free(h_M);
  free(h_N);
  free(h_P_cpu);
  free(h_P_gpu);

  return 0;
}
