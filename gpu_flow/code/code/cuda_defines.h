#ifndef CUDA_DEFINES_H
#define CUDA_DEFINES_H

#include <cuda.h>
#include <cufft.h>

//check if double precision is possible
//#if __CUDA_ARCH__<200
#ifdef PRECISION_DOUBLE
    #define CUDA_FLOAT cufftDoubleComplex
    #define CUDA_FLOAT_REAL cufftDoubleReal
    #define CUFFT_EXEC cufftExecZ2Z
    #define CUFFT_EXEC_R2C cufftExecD2Z
    #define CUFFT_EXEC_C2R cufftExecZ2D
    #define CUFFT_TYPE CUFFT_Z2Z
    #define CUFFT_TYPE_R2C CUFFT_D2Z
    #define CUFFT_TYPE_C2R CUFFT_Z2D
#else
	#define CUDA_FLOAT cufftComplex
    #define CUDA_FLOAT_REAL cufftReal
    #define CUFFT_EXEC cufftExecC2C
    #define CUFFT_EXEC_R2C cufftExecR2C
    #define CUFFT_EXEC_C2R cufftExecC2R
    #define CUFFT_TYPE CUFFT_C2C
    #define CUFFT_TYPE_R2C CUFFT_R2C
    #define CUFFT_TYPE_C2R CUFFT_C2R
#endif

// the maximum number of z modes
//define MAX_MOZ 2

//the maximum number of threads per block
#define MAX_NUMBER_THREADS_PER_BLOCK 512

//some physical constants
#define GRAVITY_CONSTANT 9.81

// path lengths
#define FILENAME_LEN 128

// number of z levels for particle tracer
// if more particles are set, code will linspace levels and interpolate
//define PARTICLE_INTERPOLATE_Z TODO: make default
#define PARTICLE_Z_NUM 128 // was: PARTICLE_INTERPOLATE_Z_NUMBER_Z_LEVELS
#define PARTICLE_Z_MAX (0.5f - 1e-3)

//define WRITE_VTK
#define WRITE_SNAPSHOT

#define EXIT_ERROR(x) { fprintf(stderr, "Error at %s:%d: %s\n", __FILE__, __LINE__, x); exit(EXIT_FAILURE);}
#define EXIT_ERROR2(x,y) { fprintf(stderr, "Error at %s:%d: %s %s\n", __FILE__, __LINE__, x, y); exit(EXIT_FAILURE);}
#define DBGOUT(x) { fprintf(stderr, "Debug out at %s:%d: %s\n", __FILE__, __LINE__, x);}
#define DBGSYNC() { cudaThreadSynchronize(); cudaError_t cudaError = cudaGetLastError(); if(cudaSuccess != cudaError) EXIT_ERROR2("CUDA Error", ::cudaGetErrorString(cudaError))}
#define DBGMEM()  { size_t avail, total; cudaMemGetInfo(&avail, &total); size_t used = total - avail; fprintf(stdout, "Device memory used: %ld of %ld (%d percent)\n", long(used), long(total), int(100*used/total));}

#endif


