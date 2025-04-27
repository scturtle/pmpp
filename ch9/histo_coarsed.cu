#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_BINS 256
#define CFACTOR 4

__global__ void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
  __shared__ unsigned int histo_s[NUM_BINS];
  for(unsigned int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x) {
    histo_s[binIdx] = 0u;
  }
  __syncthreads();

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(unsigned int i = tid * CFACTOR; i < ((tid + 1) * CFACTOR) && i < length; ++i) {
    int alphabet_position = data[i] - 'a';
    if(alphabet_position >= 0 && alphabet_position < 26) {
      atomicAdd(&(histo_s[alphabet_position / 4]), 1);
    }
  }
  __syncthreads();

  for(unsigned int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x) {
    unsigned int binValue = histo_s[binIdx];
    if(binValue > 0) {
      atomicAdd(&(histo[binIdx]), binValue);
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
  unsigned int length = 1024 * 1024;
  unsigned int data_size = length * sizeof(char);
  unsigned int histo_size = NUM_BINS * sizeof(unsigned int);

  char* h_data = (char*)malloc(data_size);
  unsigned int* h_histo = (unsigned int*)malloc(histo_size);
  unsigned int* h_histo_cpu = (unsigned int*)malloc(histo_size);

  srand(time(NULL));
  for (unsigned int i = 0; i < length; ++i) {
    h_data[i] = 'a' + (rand() % 26);
  }
   for (unsigned int i = 0; i < NUM_BINS; ++i) {
     h_histo[i] = 0;
   }

  char* d_data;
  unsigned int* d_histo;
  cudaMalloc((void**)&d_data, data_size);
  cudaMalloc((void**)&d_histo, histo_size);

  cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
  cudaMemset(d_histo, 0, histo_size);

  unsigned int num_threads = 256;
  unsigned int elements_per_block = num_threads * CFACTOR;
  unsigned int num_blocks = (length + elements_per_block - 1) / elements_per_block;

  histo_private_kernel<<<num_blocks, num_threads>>>(d_data, length, d_histo);

  cudaDeviceSynchronize();

  cudaMemcpy(h_histo, d_histo, histo_size, cudaMemcpyDeviceToHost);

  histo_cpu(h_data, length, h_histo_cpu);

  int errors = 0;
  for (unsigned int i = 0; i < NUM_BINS; ++i) {
    if (h_histo[i] != h_histo_cpu[i]) {
       errors++;
    }
  }

  if (errors == 0) {
    printf("Verification PASSED\n");
  } else {
    printf("Verification FAILED (%d errors)\n", errors);
  }

  cudaFree(d_data);
  cudaFree(d_histo);
  free(h_data);
  free(h_histo);
  free(h_histo_cpu);

  return 0;
}
