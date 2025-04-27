#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vecAddKernel(float* A_d, float* B_d, float* C_d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C_d[i] = A_d[i] + B_d[i];
  }
}

void vecAddCPU(float* A_h, float* B_h, float* C_ref, int n) {
  for (int i = 0; i < n; ++i) {
    C_ref[i] = A_h[i] + B_h[i];
  }
}

int main() {
  const int n = 1000000;
  const size_t bytes = n * sizeof(float);

  float *A_h, *B_h, *C_h, *C_ref;
  A_h = (float*)malloc(bytes);
  B_h = (float*)malloc(bytes);
  C_h = (float*)malloc(bytes);
  C_ref = (float*)malloc(bytes);

  for (int i = 0; i < n; ++i) {
    A_h[i] = sinf(i) * 2.5f;
    B_h[i] = cosf(i) * 1.5f;
  }

  float *A_d, *B_d, *C_d;
  cudaMalloc((void**)&A_d, bytes);
  cudaMalloc((void**)&B_d, bytes);
  cudaMalloc((void**)&C_d, bytes);

  cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

  cudaDeviceSynchronize();

  cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

  vecAddCPU(A_h, B_h, C_ref, n);

  float epsilon = 1e-5;
  bool success = true;
  for (int i = 0; i < n; ++i) {
    if (fabsf(C_h[i] - C_ref[i]) > epsilon) {
      success = false;
      break;
    }
  }

  if (success) {
    printf("Verification PASSED!\n");
  } else {
    printf("Verification FAILED!\n");
  }

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  free(A_h);
  free(B_h);
  free(C_h);
  free(C_ref);

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
