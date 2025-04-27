#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < Width) && (col < Width)) {
    float Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[row * Width + k] * N[k * Width + col];
    }
    P[row * Width + col] = Pvalue;
  }
}

void MatrixMulCPU(float* M, float* N, float* P, int Width) {
  for (int row = 0; row < Width; ++row) {
    for (int col = 0; col < Width; ++col) {
      float Pvalue = 0;
      for (int k = 0; k < Width; ++k) {
        Pvalue += M[row * Width + k] * N[k * Width + col];
      }
      P[row * Width + col] = Pvalue;
    }
  }
}

int main() {
  const int WIDTH = 256;
  const int TILE_WIDTH = 16;
  const size_t matrixSizeBytes = WIDTH * WIDTH * sizeof(float);

  float *h_M, *h_N, *h_P_cpu, *h_P_gpu;
  float *d_M, *d_N, *d_P;

  h_M = (float*)malloc(matrixSizeBytes);
  h_N = (float*)malloc(matrixSizeBytes);
  h_P_cpu = (float*)malloc(matrixSizeBytes);
  h_P_gpu = (float*)malloc(matrixSizeBytes);

  for (int i = 0; i < WIDTH * WIDTH; ++i) {
    h_M[i] = (float)rand() / RAND_MAX;
    h_N[i] = (float)rand() / RAND_MAX;
  }

  cudaMalloc((void**)&d_M, matrixSizeBytes);
  cudaMalloc((void**)&d_N, matrixSizeBytes);
  cudaMalloc((void**)&d_P, matrixSizeBytes);

  cudaMemcpy(d_M, h_M, matrixSizeBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, matrixSizeBytes, cudaMemcpyHostToDevice);

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, WIDTH);
  cudaDeviceSynchronize();

  cudaMemcpy(h_P_gpu, d_P, matrixSizeBytes, cudaMemcpyDeviceToHost);

  MatrixMulCPU(h_M, h_N, h_P_cpu, WIDTH);

  float tolerance = 1e-4;
  int errors = 0;
  for (int i = 0; i < WIDTH * WIDTH; ++i) {
    if (fabs(h_P_cpu[i] - h_P_gpu[i]) > tolerance) {
      errors++;
    }
  }

  if (errors == 0) {
    printf("Verification PASSED!\n");
  } else {
    printf("Verification FAILED! (%d errors)\n", errors);
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
