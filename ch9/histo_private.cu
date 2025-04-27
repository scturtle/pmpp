#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BINS 7 // (26 + 3) / 4

__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo) {
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < length) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      atomicAdd(&(histo[blockIdx.x * NUM_BINS + alphabet_position / 4]), 1);
    }
  }
  if(blockIdx.x > 0) {
    __syncthreads();
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
      unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
      if(binValue > 0) {
        atomicAdd(&(histo[bin]), binValue);
      }
    }
  }
}

void histo_cpu(char *data, unsigned int length, unsigned int *histo_cpu) {
  memset(histo_cpu, 0, NUM_BINS * sizeof(unsigned int));
  for (unsigned int i = 0; i < length; ++i) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      histo_cpu[alphabet_position / 4]++;
    }
  }
}

int main() {
  unsigned int data_len = 1024 * 1024;
  unsigned int block_size = 256;
  unsigned int num_bins = NUM_BINS;

  unsigned int grid_size = (data_len + block_size - 1) / block_size;
  unsigned int histo_mem_size = grid_size * num_bins * sizeof(unsigned int);
  unsigned int data_mem_size = data_len * sizeof(char);

  char *h_data = (char*)malloc(data_mem_size);
  unsigned int *h_histo_gpu = (unsigned int*)malloc(num_bins * sizeof(unsigned int));
  unsigned int *h_histo_cpu = (unsigned int*)malloc(num_bins * sizeof(unsigned int));

  for (unsigned int i = 0; i < data_len; ++i) {
    h_data[i] = 'a' + (rand() % 26);
  }
  memset(h_histo_gpu, 0, num_bins * sizeof(unsigned int));

  char *d_data;
  unsigned int *d_histo;

  cudaMalloc((void**)&d_data, data_mem_size);
  cudaMalloc((void**)&d_histo, histo_mem_size);

  cudaMemcpy(d_data, h_data, data_mem_size, cudaMemcpyHostToDevice);
  cudaMemset(d_histo, 0, histo_mem_size);

  histo_private_kernel<<<grid_size, block_size>>>(d_data, data_len, d_histo);

  cudaDeviceSynchronize();

  cudaMemcpy(h_histo_gpu, d_histo, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  histo_cpu(h_data, data_len, h_histo_cpu);

  int errors = 0;
  for (unsigned int i = 0; i < num_bins; ++i) {
    if (h_histo_cpu[i] != h_histo_gpu[i]) {
      errors++;
    }
  }

  if (errors == 0) {
    printf("Verification PASSED!\n");
  } else {
    printf("Verification FAILED with %d errors.\n", errors);
  }

  cudaFree(d_histo);
  cudaFree(d_data);
  free(h_histo_cpu);
  free(h_histo_gpu);
  free(h_data);

  return 0;
}
