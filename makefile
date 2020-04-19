all: life32x32 lifeGrid

life32x32: life32x32.cu
	nvcc -o life32x32 life32x32.cu

lifeGrid: lifeGrid.cu detectComputeCapability.cu
	nvcc -o detectComputeCapability detectComputeCapability.cu -lcuda
	nvcc -arch=sm_`./detectComputeCapability` -rdc=true -o lifeGrid lifeGrid.cu -lcuda
	rm detectComputeCapability