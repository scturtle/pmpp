#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_BINS 7

__global__ void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
  __shared__ unsigned int histo_s[NUM_BINS];
  for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }
  __syncthreads();

  unsigned int accumulator = 0;
  int prevBinIdx = -1;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
    int alphabet_position = data[i] - 'a';
    if(alphabet_position >= 0 && alphabet_position < 26) {
      int bin = alphabet_position / 4;
      if(bin == prevBinIdx) {
        ++accumulator;
      } else {
        if(accumulator > 0) {
          atomicAdd(&(histo_s[prevBinIdx]), accumulator);
        }
        accumulator = 1;
        prevBinIdx = bin;
      }
    }
  }
  if(accumulator > 0) {
    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
  }
  __syncthreads();

  for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned int binValue = histo_s[bin];
    if(binValue > 0) {
      atomicAdd(&(histo[bin]), binValue);
    }
  }
}

void cpu_histogram(const char* data, unsigned int length, unsigned int* histo_cpu) {
  for (unsigned int i = 0; i < NUM_BINS; ++i) {
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
  unsigned int blockSize = 256;
  unsigned int numBlocks = 64;

  char* h_data = new char[length];
  unsigned int* h_histo_gpu = new unsigned int[NUM_BINS];
  unsigned int* h_histo_cpu = new unsigned int[NUM_BINS];

  for (unsigned int i = 0; i < length; ++i) {
    h_data[i] = 'a' + (rand() % 26);
  }
  for (unsigned int i = 0; i < NUM_BINS; ++i) {
    h_histo_gpu[i] = 0;
  }

  char* d_data;
  unsigned int* d_histo;

  cudaMalloc((void**)&d_data, length * sizeof(char));
  cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int));

  cudaMemcpy(d_data, h_data, length * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_histo, h_histo_gpu, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

  histo_private_kernel<<<numBlocks, blockSize>>>(d_data, length, d_histo);

  cudaMemcpy(h_histo_gpu, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cpu_histogram(h_data, length, h_histo_cpu);

  bool success = true;
  for (unsigned int i = 0; i < NUM_BINS; ++i) {
    if (h_histo_gpu[i] != h_histo_cpu[i]) {
      success = false;
    }
  }

  if (success) {
    printf("Verification successful!\n");
  } else {
    printf("Verification FAILED!\n");
  }

  cudaFree(d_data);
  cudaFree(d_histo);
  delete[] h_data;
  delete[] h_histo_gpu;
  delete[] h_histo_cpu;

  return 0;
}
