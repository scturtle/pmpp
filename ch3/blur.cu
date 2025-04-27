#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLUR_SIZE 1
#define W 256
#define H 256

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  if(col < w && row < h) {
    int pixVal = 0;
    int pixels = 0;

    for(int blurRow=-BLUR_SIZE; blurRow<BLUR_SIZE+1; ++blurRow) {
      for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1; ++blurCol) {
        int curRow = row + blurRow;
        int curCol = col + blurCol;

        if(curRow>=0 && curRow<h && curCol>=0 && curCol<w) {
          pixVal += in[curRow*w + curCol];
          ++pixels;
        }
      }
    }
    out[row*w + col] = (unsigned char)(pixVal/pixels);
  }
}

void blurCPU(unsigned char *in, unsigned char *out, int w, int h, int blur_size) {
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      int pixVal = 0;
      int pixels = 0;
      for (int blurRow = -blur_size; blurRow < blur_size + 1; ++blurRow) {
        for (int blurCol = -blur_size; blurCol < blur_size + 1; ++blurCol) {
          int curRow = row + blurRow;
          int curCol = col + blurCol;
          if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
            pixVal += in[curRow * w + curCol];
            pixels++;
          }
        }
      }
      out[row * w + col] = (unsigned char)(pixVal / pixels);
    }
  }
}

int main() {
  size_t imageBytes = W * H * sizeof(unsigned char);

  unsigned char *h_in, *h_out_gpu, *h_out_cpu;
  unsigned char *d_in, *d_out;

  h_in = (unsigned char*)malloc(imageBytes);
  h_out_gpu = (unsigned char*)malloc(imageBytes);
  h_out_cpu = (unsigned char*)malloc(imageBytes);

  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      h_in[i * W + j] = (unsigned char)((i * j) % 256);
    }
  }

  cudaMalloc((void**)&d_in, imageBytes);
  cudaMalloc((void**)&d_out, imageBytes);

  cudaMemcpy(d_in, h_in, imageBytes, cudaMemcpyHostToDevice);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((W + dimBlock.x - 1) / dimBlock.x, (H + dimBlock.y - 1) / dimBlock.y);

  blurKernel<<<dimGrid, dimBlock>>>(d_in, d_out, W, H);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out_gpu, d_out, imageBytes, cudaMemcpyDeviceToHost);

  blurCPU(h_in, h_out_cpu, W, H, BLUR_SIZE);

  int errors = 0;
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      if (h_out_gpu[i * W + j] != h_out_cpu[i * W + j]) {
        errors++;
      }
    }
  }

  if (errors == 0) {
    printf("Verification PASSED\n");
  } else {
    printf("Verification FAILED - %d errors\n", errors);
  }

  cudaFree(d_in);
  cudaFree(d_out);
  free(h_in);
  free(h_out_gpu);
  free(h_out_cpu);

  return 0;
}
