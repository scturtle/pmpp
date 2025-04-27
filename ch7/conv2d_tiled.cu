#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 1
#define FILTER_DIM (2 * FILTER_RADIUS + 1)
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__constant__ float F_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height) {
  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  int load_row = block_row * OUT_TILE_DIM + ty - FILTER_RADIUS;
  int load_col = block_col * OUT_TILE_DIM + tx - FILTER_RADIUS;

  if (load_row >= 0 && load_row < height && load_col >= 0 && load_col < width) {
    N_s[ty][tx] = N[load_row * width + load_col];
  } else {
    N_s[ty][tx] = 0.0f;
  }
  __syncthreads();

  int out_row = block_row * OUT_TILE_DIM + ty - FILTER_RADIUS;
  int out_col = block_col * OUT_TILE_DIM + tx - FILTER_RADIUS;

  if (out_row >= 0 && out_row < height && out_col >= 0 && out_col < width) {
    int tileRow = ty - FILTER_RADIUS;
    int tileCol = tx - FILTER_RADIUS;
    if (tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 && tileCol < OUT_TILE_DIM) {
      float Pvalue = 0.0f;
      for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
          Pvalue += F_c[fRow][fCol] * N_s[ty - FILTER_RADIUS + fRow][tx - FILTER_RADIUS + fCol];
        }
      }
      P[out_row * width + out_col] = Pvalue;
    }
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

  dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
  dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

  convolution_tiled_2D_const_mem_kernel<<<gridDim, blockDim>>>(d_N, d_P, width, height);
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
