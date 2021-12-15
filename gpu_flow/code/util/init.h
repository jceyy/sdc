#ifndef INIT_H
#define INIT_H

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "../cuda_defines.h"
#include "../matrix/matrix_device.h"

class matrix_device_init
{
public:
    enum initType { zeros, random, dislocation, rectangle, isr, sv, cr, giantspiral, test};

    static void init_zeros      (matrix_device* to_init);
    static void init_random     (matrix_device* to_init, int seed, CUDA_FLOAT_REAL max_amplitude = 1e-6);
    static void init_dislocation(matrix_device* theta, int x_mode, int y_mode, CUDA_FLOAT_REAL max_amplitude = 1e-6);
    static void init_rectangle  (matrix_device* theta, int x_mode, int y_mode, CUDA_FLOAT_REAL max_amplitude = 1e-6);
    static void init_isr        (matrix_device* theta, int x_mode, int y_mode, CUDA_FLOAT_REAL max_amplitude = 1e-6);
    static void init_sv         (matrix_device* theta, int y_mode, double x_frac, CUDA_FLOAT_REAL max_amplitude = 1e-6);
    static void init_cr         (matrix_device* theta, int y_mode, double x_frac, CUDA_FLOAT_REAL max_amplitude = 1e-6);
    static void init_giantspiral(matrix_device* theta, int mode, CUDA_FLOAT_REAL max_amplitude = 1e-6);
    static void init_test       (matrix_device* f, matrix_device* g, matrix_device* theta, matrix_device *F, matrix_device *G);

private:
    static void fft(CUDA_FLOAT_REAL* real_data_device_in, CUDA_FLOAT *data_device_out,
                    int num_x_real, int num_y_real);
};

__host__ static dim3 create_block_dim(int num);
__host__ static dim3 create_grid_dim(int num);
__device__ static int get_global_index();

__global__ static void init_random_on_GPU(int entries, CUDA_FLOAT* __restrict__ values, int span, int seed, CUDA_FLOAT_REAL max_amplitude);
//__global__ static void add_random_on_GPU_real(int entries, CUDA_FLOAT_REAL* __restrict__ values, int seed, CUDA_FLOAT_REAL max_amplitude);

#endif // INIT_H
