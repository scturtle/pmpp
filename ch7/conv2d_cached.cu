#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 1
#define FILTER_DIM (2 * FILTER_RADIUS + 1)
#define TILE_DIM 32

__constant__ float F_c[FILTER_DIM][FILTER_DIM];

__global__ 
void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height){
  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int col = blockIdx.x * TILE_DIM + threadIdx.x; 
    
  __shared__ float ds_N[TILE_DIM][TILE_DIM];
  if(row < height && col < width){
    ds_N[threadIdx.y][threadIdx.x] = N[row * width + col];
  }else{
    ds_N[threadIdx.y][threadIdx.x] = 0;
  }
  __syncthreads();

  if(row < height && col < width){
    float Pvalue = 0.0f;
    for(int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++){
      int ds_y = threadIdx.y - FILTER_RADIUS + fRow;
      int y = row - FILTER_RADIUS + fRow;
      for(int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++){
        int ds_x = threadIdx.x - FILTER_RADIUS + fCol;
        if(ds_x >= 0 && ds_x < TILE_DIM && ds_y >= 0 && ds_y < TILE_DIM){
          Pvalue += F[fRow][fCol] * ds_N[ds_y][ds_x];
        }else{
          // out of tile, access the global memory of N
          int x = col - FILTER_RADIUS + fCol;
          if(y >= 0 && y < height && x >= 0 && x < width){
            Pvalue += F_c[fRow][fCol] * N[y * width + x];
          }
        }
      }
    }
    P[row * width + col] = Pvalue;
  }
}

void convolutionCPU(float *N, float *P, float *F, int width, int height) {
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      float Pvalue = 0.0f;
      for (int fr = 0; fr < FILTER_DIM; ++fr) {
        for (int fc = 0; fc < FILTER_DIM; ++fc) {
          int in_r = r + fr - FILTER_RADIUS;
          int in_c = c + fc - FILTER_RADIUS;
          float N_val = 0.0f;
          if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
            N_val = N[in_r * width + in_c];
          }
          Pvalue += F[fr * FILTER_DIM + fc] * N_val;
        }
      }
      P[r * width + c] = Pvalue;
    }
  }
}

int main() {
  int width = 256;
  int height = 256;
  size_t image_bytes = width * height * sizeof(float);
  size_t filter_bytes = FILTER_DIM * FILTER_DIM * sizeof(float);

  float *h_N, *h_P, *h_P_cpu, *h_F;
  float *d_N, *d_P;

  h_N = (float *)malloc(image_bytes);
  h_P = (float *)malloc(image_bytes);
  h_P_cpu = (float *)malloc(image_bytes);
  h_F = (float *)malloc(filter_bytes);

  for (int i = 0; i < width * height; ++i) {
    h_N[i] = (float)(rand() % 100) / 100.0f;
  }

  float filter_sum = 0.0f;
  for (int i = 0; i < FILTER_DIM * FILTER_DIM; ++i) {
    h_F[i] = 1.0f;
    filter_sum += h_F[i];
  }
  for (int i = 0; i < FILTER_DIM * FILTER_DIM; ++i) {
    h_F[i] /= filter_sum;
  }

  cudaMalloc(&d_N, image_bytes);
  cudaMalloc(&d_P, image_bytes);

  cudaMemcpy(d_N, h_N, image_bytes, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(F_c, h_F, filter_bytes);

  dim3 blockDim(TILE_DIM, TILE_DIM);
  dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

  convolution_cached_tiled_2D_const_mem_kernel<<<gridDim, blockDim>>>(d_N, d_P, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(h_P, d_P, image_bytes, cudaMemcpyDeviceToHost);

  convolutionCPU(h_N, h_P_cpu, h_F, width, height);

  int errors = 0;
  for (int i = 0; i < width * height; ++i) {
    float diff = fabsf(h_P[i] - h_P_cpu[i]);
    if (diff > 1e-5) {
      errors++;
    }
  }

  if (errors > 0) {
    printf("Verification FAILED with %d errors\n", errors);
  } else {
    printf("Verification PASSED\n");
  }

  cudaFree(d_N);
  cudaFree(d_P);
  free(h_N);
  free(h_P);
  free(h_P_cpu);
  free(h_F);

  return 0;
}
