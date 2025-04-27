#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define FILTER_RADIUS 1
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

// declare filter F in constant memory
__constant__ float F[FILTER_DIM][FILTER_DIM];

__global__ void convolution_2D_constant_kernel(float *N, float *P, int width, int height) {
  int outCol = blockIdx.x*blockDim.x + threadIdx.x;
  int outRow = blockIdx.y*blockDim.y + threadIdx.y;
  if (outRow < height && outCol < width) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
      for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
        int inRow = outRow - FILTER_RADIUS + fRow;
        int inCol = outCol - FILTER_RADIUS + fCol;
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          Pvalue += F[fRow][fCol] * N[inRow*width + inCol];
        }
      }
    }
    P[outRow*width + outCol] = Pvalue;
  }
}

void convolution_2D_cpu(const float *h_N, const float *h_F, float *P, int r, int width, int height) {
  int filter_dim = 2 * r + 1;
  for (int outRow = 0; outRow < height; ++outRow) {
    for (int outCol = 0; outCol < width; ++outCol) {
      float Pvalue = 0.0f;
      for (int fRow = 0; fRow < filter_dim; ++fRow) {
        for (int fCol = 0; fCol < filter_dim; ++fCol) {
          int inRow = outRow - r + fRow;
          int inCol = outCol - r + fCol;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            Pvalue += h_F[fRow * filter_dim + fCol] * h_N[inRow * width + inCol];
          }
        }
      }
      P[outRow * width + outCol] = Pvalue;
    }
  }
}

void verify_results(const float* cpu_result, const float* gpu_result, int size) {
  const float epsilon = 1e-5f;
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    if (fabs(cpu_result[i] - gpu_result[i]) > epsilon) {
      errors++;
    }
  }
  if (errors == 0) {
    printf("Verification successful!\n");
  } else {
    printf("Verification failed with %d errors.\n", errors);
  }
}

int main() {
  int width = 256;
  int height = 256;
  int r = FILTER_RADIUS;
  int filter_dim = FILTER_DIM;

  int input_size = width * height;
  int filter_size = filter_dim * filter_dim;
  int output_size = width * height;

  size_t input_bytes = input_size * sizeof(float);
  size_t filter_bytes = filter_size * sizeof(float);
  size_t output_bytes = output_size * sizeof(float);

  float *h_N, *h_F, *h_P_cpu, *h_P_gpu;
  float *d_N, *d_P;

  h_N = (float*)malloc(input_bytes);
  h_F = (float*)malloc(filter_bytes);
  h_P_cpu = (float*)malloc(output_bytes);
  h_P_gpu = (float*)malloc(output_bytes);

  for (int i = 0; i < input_size; ++i) {
    h_N[i] = (float)(rand() % 10);
  }
  for (int i = 0; i < filter_size; ++i) {
    h_F[i] = (float)(rand() % 3);
  }

  cudaMalloc((void**)&d_N, input_bytes);
  cudaMalloc((void**)&d_P, output_bytes);

  cudaMemcpy(d_N, h_N, input_bytes, cudaMemcpyHostToDevice);

  // copy host filter h_F to __constant__ memory symbol F
  cudaMemcpyToSymbol(F, h_F, filter_bytes);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

  convolution_2D_constant_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(h_P_gpu, d_P, output_bytes, cudaMemcpyDeviceToHost);

  convolution_2D_cpu(h_N, h_F, h_P_cpu, r, width, height);

  verify_results(h_P_cpu, h_P_gpu, output_size);

  cudaFree(d_N);
  cudaFree(d_P);

  free(h_N);
  free(h_F);
  free(h_P_cpu);
  free(h_P_gpu);

  return 0;
}
