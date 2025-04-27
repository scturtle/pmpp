#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BINS 7 // (26+3)/4

__global__ void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
  __shared__ unsigned int histo_s[NUM_BINS];
  for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }
  __syncthreads();

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < length) {
    int alphabet_position = data[i] - 'a';
    if(alphabet_position >= 0 && alphabet_position < 26) {
      atomicAdd(&(histo_s[alphabet_position/4]), 1);
    }
  }
  __syncthreads();

  for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned int binValue = histo_s[bin];
    if(binValue > 0) {
      atomicAdd(&(histo[bin]), binValue);
    }
  }
}

void histo_cpu(char* data, unsigned int length, unsigned int* histo_cpu) {
  for (int i = 0; i < NUM_BINS; ++i) {
    histo_cpu[i] = 0;
  }
  for (unsigned int i = 0; i < length; ++i) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      histo_cpu[alphabet_position / 4]++;
    }
  }
}

int main() {
  unsigned int length = 1 << 20;
  size_t data_size = length * sizeof(char);
  size_t histo_size = NUM_BINS * sizeof(unsigned int);

  char* h_data = (char*)malloc(data_size);
  unsigned int* h_histo = (unsigned int*)malloc(histo_size);
  unsigned int* h_histo_cpu = (unsigned int*)malloc(histo_size);

  for (unsigned int i = 0; i < length; ++i) {
    h_data[i] = 'a' + (rand() % 26);
  }
  memset(h_histo, 0, histo_size);

  char* d_data;
  unsigned int* d_histo;

  cudaMalloc((void**)&d_data, data_size);
  cudaMalloc((void**)&d_histo, histo_size);

  cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
  cudaMemset(d_histo, 0, histo_size);

  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;

  histo_private_kernel<<<gridSize, blockSize>>>(d_data, length, d_histo);

  cudaDeviceSynchronize();

  cudaMemcpy(h_histo, d_histo, histo_size, cudaMemcpyDeviceToHost);

  histo_cpu(h_data, length, h_histo_cpu);

  int errors = 0;
  for (int i = 0; i < NUM_BINS; ++i) {
    if (h_histo[i] != h_histo_cpu[i]) {
      errors++;
    }
  }

  if (errors == 0) {
    printf("Verification successful!\n");
  } else {
    printf("Verification failed with %d errors.\n", errors);
  }

  cudaFree(d_data);
  cudaFree(d_histo);
  free(h_data);
  free(h_histo);
  free(h_histo_cpu);

  return 0;
}
