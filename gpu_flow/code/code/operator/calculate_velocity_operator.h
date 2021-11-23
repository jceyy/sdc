#ifndef CALCULATE_VELOCITY_OPERATOR_H
#define CALCULATE_VELOCITY_OPERATOR_H

#include <vector>
#include <cuda.h>
#include <cufft.h>

#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"


class calculate_velocity_operator {

public:
    static calculate_velocity_operator* init(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length);
    ~calculate_velocity_operator();

    matrix_folder_real* calculate_operator(matrix_folder* f, matrix_folder* g,
                                           matrix_folder* F, matrix_folder* G,
                                           int num_z = 5);

    matrix_folder_real* calculate_derivative_matrix(matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G);
    void calculate_operator_at(CUDA_FLOAT_REAL *at_z_device, int at_size, matrix_folder_real* derivs,
                               CUDA_FLOAT_REAL *u_1, CUDA_FLOAT_REAL *u_2, CUDA_FLOAT_REAL *u_3);
    void calculate_operator_at(CUDA_FLOAT_REAL *at_z_device, int at_size,
                               matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G,
                               CUDA_FLOAT_REAL *u_1, CUDA_FLOAT_REAL *u_2, CUDA_FLOAT_REAL *u_3);


protected:
	CUDA_FLOAT_REAL cube_length_x;
	CUDA_FLOAT_REAL cube_length_y;
	CUDA_FLOAT_REAL cube_length_z;
	
	//cuda fft plans
	cufftHandle c2r_plan;

private:
	calculate_velocity_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length);
    void calculate_derivatives(matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G,
                               CUDA_FLOAT_REAL* u_1_1, CUDA_FLOAT_REAL* u_2_1, CUDA_FLOAT_REAL* u_3_1,
                               CUDA_FLOAT_REAL* u_1_2, CUDA_FLOAT_REAL* u_2_2);
};


__host__ static dim3 create_grid_dim(long number_of_matrix_entries);
__host__ static dim3 create_block_dim(long number_of_matrix_entries);
__device__ static int get_global_index();
__device__ static void get_current_matrix_indices(int &col_index, int &row_index, int &mat_index,
                                                  int total_index, int columns, int rows, int matrices);

__global__ static void linspace(int entries, CUDA_FLOAT_REAL* positions,
                                CUDA_FLOAT_REAL min, CUDA_FLOAT_REAL max);

__global__ static void create_derivative_real_matrix_x(CUDA_FLOAT* input, CUDA_FLOAT* output,
                                                       int columns, int rows, int matrices,
                                                       CUDA_FLOAT_REAL cube_length_x,
                                                       CUDA_FLOAT_REAL factor);
__global__ static void create_derivative_real_matrix_y(CUDA_FLOAT* input, CUDA_FLOAT* output,
                                                       int columns, int rows, int matrices,
                                                       CUDA_FLOAT_REAL cube_length_y,
                                                       CUDA_FLOAT_REAL factor);
__global__ static void create_laplace_real_matrix(CUDA_FLOAT* input, CUDA_FLOAT* output,
                                             int columns, int rows, int matrices,
                                             CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                                  CUDA_FLOAT_REAL factor);
__global__ static void compose_u_from_derivs(CUDA_FLOAT_REAL* u_1_1, CUDA_FLOAT_REAL* u_2_1, CUDA_FLOAT_REAL* u_3_1,
                                             CUDA_FLOAT_REAL* u_1_2, CUDA_FLOAT_REAL* u_2_2,
                                        CUDA_FLOAT_REAL* Ci, CUDA_FLOAT_REAL* Si, CUDA_FLOAT_REAL* dzCi, int num_ansatz,
                                        int xy_size, int z_size, CUDA_FLOAT_REAL* u_1, CUDA_FLOAT_REAL* u_2, CUDA_FLOAT_REAL* u_3);
__global__ static void get_z_coeff_at(CUDA_FLOAT_REAL* at_z, int z_size, int num_ansatz, CUDA_FLOAT_REAL* Ci, CUDA_FLOAT_REAL* Si, CUDA_FLOAT_REAL* dzCi);

__global__ static void add_dc_part(CUDA_FLOAT* g, CUDA_FLOAT* F,
                              int columns, int rows, int matrices);





#endif



