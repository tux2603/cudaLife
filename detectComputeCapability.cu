
#include <cuda.h>
#include <iostream>

int main() {
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("%d%d\n", deviceProp.major, deviceProp.minor);
}