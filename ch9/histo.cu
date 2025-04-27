#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define HISTO_BINS 7 // (26 + 3) / 4

__global__ void histo_kernel(char *data, unsigned int length, unsigned int *histo) {
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < length) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      atomicAdd(&(histo[alphabet_position/4]), 1);
    }
  }
}

void histo_cpu(char *data, unsigned int length, unsigned int *histo_cpu) {
  for (int i = 0; i < HISTO_BINS; ++i) {
    histo_cpu[i] = 0;
  }
  for (unsigned int i = 0; i < length; ++i) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      histo_cpu[alphabet_position/4]++;
    }
  }
}

int main() {
  unsigned int length = 1024 * 1024;
  size_t data_size = length * sizeof(char);
  size_t histo_size_bytes = HISTO_BINS * sizeof(unsigned int);

  char *h_data = (char*)malloc(data_size);
  unsigned int *h_histo_gpu = (unsigned int*)malloc(histo_size_bytes);
  unsigned int *h_histo_cpu = (unsigned int*)malloc(histo_size_bytes);

  for (unsigned int i = 0; i < length; ++i) {
    h_data[i] = 'a' + (rand() % 26);
  }
  for (int i = 0; i < HISTO_BINS; ++i) {
    h_histo_gpu[i] = 0;
  }

  char *d_data = NULL;
  unsigned int *d_histo = NULL;
  cudaMalloc((void**)&d_data, data_size);
  cudaMalloc((void**)&d_histo, histo_size_bytes);

  cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_histo, h_histo_gpu, histo_size_bytes, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;

  histo_kernel<<<gridSize, blockSize>>>(d_data, length, d_histo);

  cudaMemcpy(h_histo_gpu, d_histo, histo_size_bytes, cudaMemcpyDeviceToHost);

  histo_cpu(h_data, length, h_histo_cpu);

  int errors = 0;
  for (int i = 0; i < HISTO_BINS; ++i) {
    if (h_histo_cpu[i] != h_histo_gpu[i]) {
      errors++;
    }
  }

  if (errors == 0) {
    printf("Verification PASSED\n");
  } else {
    printf("Verification FAILED: %d errors\n", errors);
  }

  free(h_data);
  free(h_histo_gpu);
  free(h_histo_cpu);
  cudaFree(d_data);
  cudaFree(d_histo);

  return 0;
}
