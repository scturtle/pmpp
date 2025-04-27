#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHANNELS 3

__global__
void colortoGrayscaleConvertion(unsigned char * Pout,
                                unsigned char * Pin, int width, int height) {
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    int grayOffset = row*width + col;
    int rgbOffset = grayOffset*CHANNELS;
    unsigned char r = Pin[rgbOffset + 0];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];
    Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
  }
}


void cpuGrayscaleConversion(unsigned char * Pout, unsigned char * Pin, int width, int height) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int grayOffset = row * width + col;
      int rgbOffset = grayOffset * CHANNELS;
      unsigned char r = Pin[rgbOffset + 0];
      unsigned char g = Pin[rgbOffset + 1];
      unsigned char b = Pin[rgbOffset + 2];
      float grayVal = 0.21f * r + 0.71f * g + 0.07f * b;
      Pout[grayOffset] = (unsigned char)grayVal;
    }
  }
}

int main() {
  int width = 512;
  int height = 512;

  size_t rgb_size = width * height * CHANNELS * sizeof(unsigned char);
  size_t gray_size = width * height * sizeof(unsigned char);

  unsigned char *h_Pin, *h_Pout_gpu, *h_Pout_cpu;
  unsigned char *d_Pin, *d_Pout;

  h_Pin = (unsigned char*)malloc(rgb_size);
  h_Pout_gpu = (unsigned char*)malloc(gray_size);
  h_Pout_cpu = (unsigned char*)malloc(gray_size);

  for (int i = 0; i < width * height * CHANNELS; ++i) {
    h_Pin[i] = (unsigned char)(rand() % 256);
  }

  cudaMalloc((void**)&d_Pin, rgb_size);
  cudaMalloc((void**)&d_Pout, gray_size);

  cudaMemcpy(d_Pin, h_Pin, rgb_size, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  colortoGrayscaleConvertion<<<gridSize, blockSize>>>(d_Pout, d_Pin, width, height);

  cudaMemcpy(h_Pout_gpu, d_Pout, gray_size, cudaMemcpyDeviceToHost);

  cpuGrayscaleConversion(h_Pout_cpu, h_Pin, width, height);

  int errors = 0;
  for (int i = 0; i < width * height; ++i) {
    if (h_Pout_cpu[i] != h_Pout_gpu[i]) {
      errors++;
    }
  }

  if (errors == 0) {
    printf("Verification successful!\n");
  } else {
    printf("Verification failed! Number of errors: %d\n", errors);
  }

  cudaFree(d_Pout);
  cudaFree(d_Pin);
  free(h_Pout_cpu);
  free(h_Pout_gpu);
  free(h_Pin);

  return 0;
}
