#ifndef FREEZE_OPERATOR_H
#define FREEZE_OPERATOR_H

//system includes
#include <vector>
#include <cuda.h>
#include <cufft.h>

//my defines
#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"


class freeze_operator {


private:
    freeze_operator(std::vector<int> dimension);

    void freeze_mask(matrix_folder* theta, matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G);
	

protected:

	//cuda fft plans
	cufftHandle c2r_plan;
    cufftHandle r2c_plan;

    bool frozen;
    int* mask_frozen;
    matrix_device_real* theta_frozen;
    matrix_device_real* f_frozen;
    matrix_device_real* g_frozen;

public:
    ~freeze_operator();

	//init the operator
    static freeze_operator* init_operator(std::vector<int> dimension);

    void freeze_circle(matrix_folder* theta, matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G, CUDA_FLOAT_REAL radius);
    void freeze_rectangle(matrix_folder* theta, matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G, CUDA_FLOAT_REAL width);

    // calculate frozen operator
    void calculate_operator(matrix_folder* theta, matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G);

};

__host__ static dim3 create_grid_dim(int num);
__host__ static dim3 create_block_dim(int num);

//helper functions
__device__ static int get_global_index();
//to create column, row and matrix index from a global index
__device__ static void get_current_matrix_indices(int &current_col, int &current_row, int &current_matrix, int total_index, int columns, int rows, int matrices);

__global__ static void apply_mask(CUDA_FLOAT_REAL* input_output, CUDA_FLOAT_REAL* mask_data, int* mask, CUDA_FLOAT_REAL factor, int num_xy, int num_entries);
__global__ static void init_mask_physical_space_rectangle(int* mask, int columns, int rows, CUDA_FLOAT_REAL width);
__global__ static void init_mask_physical_space_circle(int* mask, int columns, int rows, CUDA_FLOAT_REAL radius);

#endif















