// test_cuda_consle_2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"


#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
int main()
{

	int deviceCount = 0;
	
	cudaGetDeviceCount(&deviceCount);//runtime API中的函数以cuda为前缀，driver API中的函数则以cu为前缀
	if (deviceCount == 0)
	{
		printf("There is no device suppporting CUDA\n");
	}
	int dev = 0;
	for (; dev < deviceCount; dev++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp,dev);
		if (dev == 0)
		{
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)//deviceProp.(major,minor)分别是设备计算能力的主版本号和次版本号
				printf("There is no device supporting CUDA.\n");
			else if (deviceCount == 1)
				printf("There is 1 device supporting CUDA.\n");
			else
				printf("there are %d devices supporting CUDA.\n",deviceCount);
		}
		printf("\n Device %d:\"%s\"\n",dev,deviceProp.name);
		printf("Major revision number: %d\n",deviceProp.major);
		printf("Minor revision number: %d\n",deviceProp.minor);
		printf("Total amount of global memory: %u bytes\n",deviceProp.totalGlobalMem);

#if CUDART_VERSION>=2000
		printf("Number of multiprocessors:%d\n",deviceProp.multiProcessorCount);
		printf("Number of cores:%d\n",8*deviceProp.multiProcessorCount);
#endif
		printf("Total amount of constant memory:%u bytes\n",deviceProp.totalConstMem);
		printf("Total amount of shared memory per block: %u bytes\n",deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:%d\n",deviceProp.regsPerBlock);
		printf("Warp size:%d\n",deviceProp.warpSize);
		printf("Maximum number of threads per block: %d\n",deviceProp.maxThreadsPerBlock);
		printf("Maximum sizes of each dimension of a block:%d x %d x %d \n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[1]);
		printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch: %u bytes\n",deviceProp.memPitch);
		printf("Texture alignment: %u bytes\n",deviceProp.textureAlignment);
		printf("Clock rate: %.2f GHz\n",deviceProp.clockRate*1e-6);
#if CUDART_VERSION>=2000
		printf("Concurrent copy and execution:%s\n",deviceProp.deviceOverlap?"Yes":"No");
#endif


	}
	printf("\n Text PASSED\n");


	const int arraySize = 5;
	const int a[arraySize] = { 1,2,3,4,5 };
	const int b[arraySize] = { 10,20,30,40,50 };
	int c[arraySize] = { 0 };
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed");
		return 1;
	}

	printf("{1,2,3,4,5}+{10,20,30,40,50}={%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);
	printf("cuda工程中调用cpp成功");

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed");
		return 1;
	}
	getchar();


	return 0;
}