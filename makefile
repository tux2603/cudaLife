all: life32x32 lifeGrid

life32x32: life32x32.cu
	nvcc -o life32x32 life32x32.cu

lifeGrid: lifeGrid.cu detectComputeCapability.cu
	#nvcc -o detectComputeCapability detectComputeCapability.cu -lcuda
	#@echo "Detecting compute capability of the first available device"
	#$(eval COMPUTE=$(shell ./detectComputeCapability))
	#rm detectComputeCapability
	nvcc -arch=sm_75 -rdc=true -o lifeGrid lifeGrid.cu -lcuda