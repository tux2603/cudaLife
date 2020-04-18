/*
 * The kernel found in life32x32.cu modified to handle an array of blocks
 * distributed over a grid. Note that the kernel uses bitmasks to perform the
 * modulo operations to facilitae wrap around, so it is only possible to have
 * grids with power of two dimensions.
 */

#include <chrono>
#include <cooperative_groups.h>
#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

#define GRID_X_LOG2 2
#define GRID_Y_LOG2 2

#define SLEEP_TIME 50000

#define GENERATION_STEP 1

const char* outputVector[4] = {(char*)" ", (char*)"▀", (char*)"▄", (char*)"█"};

__global__ void tiledLifeKernel(uint *cols, uint32_t numgenerations,
                                uint16_t gridDimXLog2, uint16_t gridDimYLog2) {
  // Allocate the local shared memory needed for the kernel. The memory will be
  // arranged in a 34x34 array, with a one byte padding at the end of every row
  // to prevent bank conflicts
  __shared__ uint32_t cells[1190];
  __shared__ uint32_t sidesData[2];

  // Get the grid group that will be used to synchronize the entire kernel
  // between generations
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Got the indexes needed to read the adjacent blocks, allowing for wrap
  // around
  uint32_t sideBlockX[2];
  sideBlockX[0] = (blockIdx.x - 1) & (0xffffffff >> (32 - gridDimXLog2));
  sideBlockX[1] = (blockIdx.x + 1) & (0xffffffff >> (32 - gridDimXLog2));
  uint32_t topBlockY = (blockIdx.y - 1) & (0xffffffff >> (32 - gridDimYLog2));
  uint32_t bottomBlockY = (blockIdx.y + 1) & (0xffffffff >> (32 - gridDimYLog2));

  // Load the col index to a register
  // TODO Hopefully won't run out of registers, but might want to banchmark this
  // anyway
  uint16_t colIdx = threadIdx.x;

  // Load the data in the block itself
  uint32_t colData =
      cols[(((blockIdx.y << gridDimXLog2) + blockIdx.x) << 5) + colIdx];

  for (uint8_t i = 0; i < 32; ++i) {
    cells[(i + 1) * 35 + colIdx + 1] = (colData >> i) & 0x1;
  }

  // Begin performing the actual cellular automation portion of the kernel
  for (uint32_t g = 0; g < numgenerations; ++g) {
    // Each generation will need to reload all the neighboring cells

    // #######################################################################
    // #####                    BEGIN WARP DIVERGENCE                    #####
    // #######################################################################

    // Read the four corners of adjacent to the block and sides of the block
    // Note that this will cause warp divergence, but I can't think of a way to
    // avoid it right now
    if (colIdx < 2) {
      uint8_t offset = 0x1f * (~colIdx & 0x1);

      // Read amd save the top corners
      uint32_t colIn =
          cols[(((topBlockY << gridDimXLog2) + sideBlockX[colIdx]) << 5) +
               offset];
      cells[colIdx * 33] = colIn >> 31;

      // Read the sides, they will be saved later by individual threads
      sidesData[colIdx] =
          cols[(((blockIdx.y << gridDimXLog2) + sideBlockX[colIdx]) << 5) +
               offset];

      // Read and saved the bottom corners
      colIn = 
          cols[(((bottomBlockY << gridDimXLog2) + sideBlockX[colIdx]) << 5) +
               offset];
      cells[33 * 35 + colIdx * 33] = colIn & 0b1;
    }

    // #######################################################################
    // #####                     END WARP DIVERGENCE                     #####
    // #######################################################################

    // Now that that divergent mess is done, sync everything back up
    // TODO: figure out if this is strictly necessary
    __syncthreads();

    // The left and right sides were loaded in in the divergent section, but
    // they can be transfered to the memory array in parallel

    // Store the left side to shared memory
    uint32_t leftSide = sidesData[0];
    cells[(colIdx + 1) * 35] = (leftSide >> colIdx) & 0x1;

    // Store the right side to shared memory
    uint32_t rightSide = sidesData[1];
    cells[(colIdx + 1) * 35 + 33] = (rightSide >> colIdx) & 0x1;

    // Now we can just load the rest of the data in by column

    // Load in the neighbor above this column
    uint32_t colData = 
        cols[(((topBlockY << gridDimXLog2) + blockIdx.x) << 5) + colIdx];
    cells[colIdx + 1] = (colData >> 31) & 0x1;

    // Load in the neighbors below this column
    colData = 
        cols[(((bottomBlockY << gridDimXLog2) + blockIdx.x) << 5) + colIdx];
    cells[33 * 35 + colIdx + 1] = colData & 0x1;

    // #######################################################################
    // ######                     END MEMORY LOADING                    ######
    // #######################################################################

    uint8_t lastSides = 0, lastMiddle = 0, thisSides = 0, thisMiddle = 0,
            nextSides = 0, nextMiddle = 0;

    // Get the neighbors in the previoous row
    lastSides = cells[colIdx] & 0x1;
    lastSides += cells[colIdx + 2] & 0x1;
    lastMiddle = cells[colIdx + 1] & 0x1;

    // Get the neighbors in the current row
    thisSides = cells[35 + colIdx] & 0x1;
    thisSides += cells[35 + colIdx + 2] & 0x1;

    // Get the state of the current cell
    thisMiddle = cells[35 + colIdx + 1] & 0x1;

    for (int i = 1; i < 33; ++i) {
      // Get the neighbors in the next row
      nextSides = cells[(i + 1) * 35 + colIdx] & 0x1;
      nextSides += cells[(i + 1) * 35 + colIdx + 2] & 0x1;
      nextMiddle = cells[(i + 1) * 35 + colIdx + 1] & 0x1;

      // Compute the total number of neighbors
      uint8_t neighbors =
          lastSides + lastMiddle + thisSides + nextSides + nextMiddle;

      // Compute the next state of the cell
      cells[i * 35 + colIdx + 1] |= 
          (~neighbors >> 1 & neighbors & (thisMiddle | neighbors) << 1) & 0x2;

      // The current row will becom the next row, etc
      lastSides = thisSides;
      lastMiddle = thisMiddle;
      thisSides = nextSides;
      thisMiddle = nextMiddle;
    }

    // #######################################################################
    // #####                  END CELLULAR COMPUTATIONS                  #####
    // #######################################################################

    // Make sure all the threads have finished computing before continuing to
    // the rest of the memory management
    __syncthreads();

    // Shift the next state of the cell into the current state of the cell
    for (int i = 1; i < 33; ++i) {
      cells[i * 35 + colIdx + 1] >>= 1;
    }

    // Write back the computed column data
    // First, clear the register that will hold the compressed data
    colData = 0x00000000;

    // Then compress all the data into the register
    for (int i = 0; i < 32; ++i) {
      colData |= (cells[(i+1) * 35 + colIdx + 1] & 0x1) << i;
    }

    // Write the column data back to global memory so that other blocks can
    // access it
    cols[(((blockIdx.y << gridDimXLog2) + blockIdx.x) << 5) + colIdx] = colData;

    // Synchronize all blocks in the kernel before starting on the next
    // generation
    grid.sync();
  }

  // #########################################################################
  // #####                        END LIFE KERNEL                        #####
  // #########################################################################
}

void generateCells(uint32_t *cells, uint32_t length) {
  uint32_t seed = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  srand(seed);

  for (uint32_t i = 0; i < length; ++i) {
    cells[i] = (rand() + rand() + rand()) & (rand() + rand() + rand()) & 0xFFFFFFFF;
  }
}

void drawCells(uint32_t *cells, int generation, uint32_t sleepTime, uint16_t gridDimXLog2,
               uint16_t gridDimYLog2) {
  printf("\033[H");

  for (int b = 0; b < 1 << gridDimYLog2; ++b) {
    for (int y = 0; y < 32; y += 2) {
      // printf("\n\033[1;%dH", y+1);
      for (int x = 0; x < (1 << gridDimXLog2) * 32; ++x) {
        uint8_t cellsInBlock = 0;
        cellsInBlock = (cells[b * (1 << (gridDimXLog2 + 5)) + x] & (0x3l << y)) >> y;
        // bool lowerCell = (cells[b * (1 << (gridDimXLog2 + 5)) + x] & (0x1l << (y+1)) >> (y+1);
        // cellsInBlock |= 0x3 & (cells[b * (1 << (gridDimXLog2 + 5)) + x] & (0x1ull << y)) >> y;
        printf(outputVector[cellsInBlock]);

      }
      printf("\n");
    }
  }
  printf("%d    ", generation);
  usleep(sleepTime);
}
void launchKernel(uint32_t *cells, uint32_t numGenerations,
                  uint16_t gridDimXLog2, uint16_t gridDimYLog2) {
  dim3 gridDim;
  gridDim.x = 1 << gridDimXLog2;
  gridDim.y = 1 << gridDimYLog2;
  gridDim.z = 1;

  dim3 blockDim;
  blockDim.x = 32;
  blockDim.y = 1;
  blockDim.z = 1;

  void **args = new void *[4];
  args[0] = &cells;
  args[1] = &numGenerations;
  args[2] = &gridDimXLog2;
  args[3] = &gridDimYLog2;

  cudaLaunchCooperativeKernel((void *)tiledLifeKernel, gridDim, blockDim, args);
  cudaDeviceSynchronize();
}

int main(int argc, char **argv) {
  // Make sure that the CUDA device has compute capability >=6.0
  // int supportsCooperativeLaunch = 0;
  // CUdevice dev;
  // cuDeviceGet(&dev, 0);
  // cudaDeviceProp deviceProp;
  // cudaGetDeviceProperties(&deviceProp, dev);
  // cuDeviceGetAttribute(&supportsCooperativeLaunch,
  //                      CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev);

  // if (supportsCooperativeLaunch == 0) {
  //   std::cout
  //       << "Your CUDA device does not support the cooperative launch feature "
  //       << "required by this program (compute capability >=6.0 required, "
  //       << deviceProp.major << "." << deviceProp.minor << " detected)"
  //       << std::endl;
  //   return 1;
  // }

  uint32_t *cells;

  cudaMallocManaged(&cells, sizeof(uint32_t) * 32 * (1<<GRID_X_LOG2) * (1<<GRID_Y_LOG2));
  generateCells(cells, 32 * (1<<GRID_X_LOG2) * (1<<GRID_Y_LOG2));

  drawCells(cells, 0, SLEEP_TIME, GRID_X_LOG2, GRID_Y_LOG2);

  uint32_t generation = 1;
  while(generation < 5 || true) {
    launchKernel(cells, GENERATION_STEP, GRID_X_LOG2, GRID_Y_LOG2);
    drawCells(cells, generation, SLEEP_TIME, GRID_X_LOG2, GRID_Y_LOG2);
    generation += GENERATION_STEP;
    
  }

  cudaFree(cells);
}