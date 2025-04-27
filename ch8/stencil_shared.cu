#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define OUT_TILE_DIM 16
#define R 1
#define IN_TILE_DIM (OUT_TILE_DIM + 2 * R)

const float c0 = 0.5f;
const float c_neighbor = (1.0f - c0) / 6.0f;
const float c1 = c_neighbor;
const float c2 = c_neighbor;
const float c3 = c_neighbor;
const float c4 = c_neighbor;
const float c5 = c_neighbor;
const float c6 = c_neighbor;

__global__ void stencil_kernel(const float* in, float* out, unsigned int N) {
  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - R;
  int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - R;
  int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - R;
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
  } else {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    if (threadIdx.z >= R && threadIdx.z < IN_TILE_DIM - R &&
        threadIdx.y >= R && threadIdx.y < IN_TILE_DIM - R &&
        threadIdx.x >= R && threadIdx.x < IN_TILE_DIM - R) {
      out[i * N * N + j * N + k] =
                     c0 * in_s[threadIdx.z]    [threadIdx.y]    [threadIdx.x]
                   + c1 * in_s[threadIdx.z]    [threadIdx.y]    [threadIdx.x - 1]
                   + c2 * in_s[threadIdx.z]    [threadIdx.y]    [threadIdx.x + 1]
                   + c3 * in_s[threadIdx.z]    [threadIdx.y - 1][threadIdx.x]
                   + c4 * in_s[threadIdx.z]    [threadIdx.y + 1][threadIdx.x]
                   + c5 * in_s[threadIdx.z - 1][threadIdx.y]    [threadIdx.x]
                   + c6 * in_s[threadIdx.z + 1][threadIdx.y]    [threadIdx.x];
    }
  }
}

void stencil_cpu(const float* in, float* out, unsigned int N) {
  for (unsigned int i = 1; i < N - 1; ++i) {
    for (unsigned int j = 1; j < N - 1; ++j) {
      for (unsigned int k = 1; k < N - 1; ++k) {
        unsigned int idx = i * N * N + j * N + k;
        out[idx] = c0 * in[idx]
                 + c1 * in[i * N * N + j * N + (k - 1)]
                 + c2 * in[i * N * N + j * N + (k + 1)]
                 + c3 * in[i * N * N + (j - 1) * N + k]
                 + c4 * in[i * N * N + (j + 1) * N + k]
                 + c5 * in[(i - 1) * N * N + j * N + k]
                 + c6 * in[(i + 1) * N * N + j * N + k];
      }
    }
  }
}

int main() {
  unsigned int N = 64;
  long long n_bytes = (long long)N * N * N * sizeof(float);

  float *h_in, *h_out, *h_verify;
  h_in = (float*)malloc(n_bytes);
  h_out = (float*)malloc(n_bytes);
  h_verify = (float*)malloc(n_bytes);

  for (long long i = 0; i < (long long)N * N * N; ++i) {
    h_in[i] = (float)(rand() % 100);
    h_out[i] = 0.0f;
    h_verify[i] = 0.0f;
  }

  float *d_in, *d_out;
  cudaMalloc((void**)&d_in, n_bytes);
  cudaMalloc((void**)&d_out, n_bytes);

  cudaMemcpy(d_in, h_in, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out, n_bytes, cudaMemcpyHostToDevice); // Initialize device output

  dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
  dim3 dimGrid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
               (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
               (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

  stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, n_bytes, cudaMemcpyDeviceToHost);

  stencil_cpu(h_in, h_verify, N);

  int errors = 0;
  float tolerance = 1e-5f;
  for (unsigned int i = 1; i < N - 1; ++i) {
    for (unsigned int j = 1; j < N - 1; ++j) {
      for (unsigned int k = 1; k < N - 1; ++k) {
        unsigned int idx = i * N * N + j * N + k;
        if (fabs(h_out[idx] - h_verify[idx]) > tolerance) {
          errors++;
        }
      }
    }
  }

  if (errors == 0) {
    printf("Verification PASSED\n");
  } else {
    printf("Verification FAILED with %d errors\n", errors);
  }

  cudaFree(d_in);
  cudaFree(d_out);
  free(h_in);
  free(h_out);
  free(h_verify);

  return 0;
}
