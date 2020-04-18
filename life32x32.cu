#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

__global__ void singleBlockLifeKernel(uint32_t *cols, int numGenerations) {
  __shared__ uint8_t grid[1024]; // TODO Should this be uint32_t?

  int colIdx = threadIdx.x;

  // Copy data from global memory to shared memory
  uint32_t colData = cols[colIdx];

  // Split the data out into an easy to handle array
  for (int i = 0; i < 32; ++i) {
    grid[i * 32 + colIdx] = ((colData & 1 << i)) >> i;
  }

  // The bit mask is a quick and dirty way of computing the positive bounded %32
  uint8_t leftIdx = ((colIdx - 1) & 0x1f);
  uint8_t rightIdx = ((colIdx + 1) & 0x1f);

  for (int g = 0; g < numGenerations; ++g) {
    uint8_t lastSides = 0, lastMiddle = 0, thisSides = 0, thisMiddle = 0,
            nextSides = 0, nextMiddle = 0;

    lastSides = grid[31 * 32 + leftIdx] & 1;
    lastSides += grid[31 * 32 + rightIdx] & 1;
    lastMiddle = grid[31 * 32 + colIdx];

    thisSides = grid[leftIdx] & 1;
    thisSides += grid[rightIdx] & 1;
    thisMiddle = grid[colIdx];

    // Perform cellular automata
    for (int i = 0; i < 31; ++i) {

      nextSides = grid[(i + 1) * 32 + leftIdx] & 1;
      nextSides += grid[(i + 1) * 32 + rightIdx] & 1;
      nextMiddle = grid[(i + 1) * 32 + colIdx];

      uint8_t neighbors =
          lastSides + lastMiddle + thisSides + nextSides + nextMiddle;

      grid[i * 32 + colIdx] |=
          (~neighbors >> 1 & neighbors & (thisMiddle | neighbors) << 1) & 2;

      lastSides = thisSides;
      lastMiddle = thisMiddle;
      thisSides = nextSides;
      thisMiddle = nextMiddle;
    }

    nextSides = grid[leftIdx] & 1;
    nextSides += grid[rightIdx] & 1;
    nextMiddle = grid[colIdx] & 1;

    uint8_t neighbors =
        lastSides + lastMiddle + thisSides + nextSides + nextMiddle;

    grid[31 * 32 + colIdx] |=
        (~neighbors >> 1 & neighbors & (thisMiddle | neighbors) << 1) & 2;

    // for(int i = 0; i < 32; ++i) {
    //   grid[i*32 + colIdx] |= ~grid[i*32 + leftIdx] << 1;
    // }
  }

  colData = 0;

  // Cram the data back into a single value
  for (int i = 0; i < 32; ++i) {
    colData |= ((grid[i * 32 + colIdx] >> 1) & 1) << i;
  }

  // Copy the data back into global memory
  cols[colIdx] = colData;
}

void generateGrid(uint32_t *&cols) {

  uint32_t seed = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  srand(seed);

  for (int i = 0; i < 32; ++i) {
    cols[i] = rand() & 0xFFFFFFFF;
  }
}

void drawGrid(uint32_t *col, int generation) {
  printf("\033[H");
  for (int y = 0; y < 32; ++y) {
    // printf("\n\033[1;%dH", y+1);
    for (int x = 0; x < 32; ++x)
      printf((col[x] & (1l << y)) ? "██" : "  ");
    printf("\n");
  }
  printf("%d    ", generation);
  usleep(50000);
}

int main() {
  uint32_t *cols;
  uint32_t generation = 0;

  cudaMallocManaged(&cols, sizeof(uint32_t) * 32);

  generateGrid(cols);
  drawGrid(cols, generation);

  while (true) {
    singleBlockLifeKernel<<<1, 32>>>(cols, 1);
    generation += 1;
    cudaDeviceSynchronize();
    drawGrid(cols, generation);
  }
  cudaFree(cols);
  return 0;
}