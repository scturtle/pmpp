#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

__global__ void matrixMulKernel(float* M, float* N, float* P, int Width, int TILE_WIDTH) {
  extern __shared__ float shared_mem[];

  float *Mds = shared_mem;
  float *Nds = shared_mem + TILE_WIDTH * TILE_WIDTH;

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Pvalue = 0;

  int Mds_tile_idx = ty * TILE_WIDTH + tx;
  int Nds_tile_idx = ty * TILE_WIDTH + tx;

  for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {

    if (Row < Width && (ph * TILE_WIDTH + tx) < Width) {
      Mds[Mds_tile_idx] = M[Row * Width + ph * TILE_WIDTH + tx];
    } else {
      Mds[Mds_tile_idx] = 0.0f;
    }
    if ((ph * TILE_WIDTH + ty) < Width && Col < Width) {
      Nds[Nds_tile_idx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
    } else {
      Nds[Nds_tile_idx] = 0.0f;
    }
    __syncthreads();

    if (Row < Width && Col < Width) {
      for (int k = 0; k < TILE_WIDTH; ++k) {
        Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
      }
    }
    __syncthreads();
  }

  if (Row < Width && Col < Width) {
    P[Row * Width + Col] = Pvalue;
  }
}

void matrixMulCPU(float* M, float* N, float* P, int Width) {
  for (int i = 0; i < Width; ++i) {
    for (int j = 0; j < Width; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < Width; ++k) {
        sum += M[i * Width + k] * N[k * Width + j];
      }
      P[i * Width + j] = sum;
    }
  }
}

int main() {
  int Width = 256;
  const int TILE_WIDTH = 16;
  assert(Width % TILE_WIDTH == 0);

  int Size = Width * Width;
  size_t bytes = Size * sizeof(float);

  float* h_M;
  float* h_N;
  float* h_P_gpu;
  float* h_P_cpu;

  h_M = (float*)malloc(bytes);
  h_N = (float*)malloc(bytes);
  h_P_gpu = (float*)malloc(bytes);
  h_P_cpu = (float*)malloc(bytes);

  for (int i = 0; i < Size; ++i) {
    h_M[i] = (float)(rand() % 100) / 10.0f;
    h_N[i] = (float)(rand() % 100) / 10.0f;
  }

  float* d_M;
  float* d_N;
  float* d_P;

  cudaMalloc(&d_M, bytes);
  cudaMalloc(&d_N, bytes);
  cudaMalloc(&d_P, bytes);

  cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);

  size_t Mds_bytes = TILE_WIDTH * TILE_WIDTH * sizeof(float);
  size_t Nds_bytes = TILE_WIDTH * TILE_WIDTH * sizeof(float);
  size_t shared_mem_size = Mds_bytes + Nds_bytes;

  matrixMulKernel<<<dimGrid, dimBlock, shared_mem_size>>>(d_M, d_N, d_P, Width, TILE_WIDTH);
  cudaDeviceSynchronize();

  cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);

  matrixMulCPU(h_M, h_N, h_P_cpu, Width);

  double diff = 0.0;
  for(int i=0; i < Size; ++i) {
    diff += std::fabs(h_P_gpu[i] - h_P_cpu[i]);
  }
  double avg_diff = diff / Size;
  double tolerance = 1e-5;

  if (avg_diff < tolerance) {
    std::cout << "Verification PASSED (Average difference: " << avg_diff << ")" << std::endl;
  } else {
    std::cout << "Verification FAILED (Average difference: " << avg_diff << ")" << std::endl;
  }

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
  free(h_M);
  free(h_N);
  free(h_P_gpu);
  free(h_P_cpu);

  return 0;
}
