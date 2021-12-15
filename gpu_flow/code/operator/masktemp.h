#ifndef MASKTEMP_H
#define MASKTEMP_H

//system includes
#include <vector>
#include <cuda.h>
#include <cufft.h>

//my defines
#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"


class masktemp {


private:
    masktemp(std::vector<int> dimension);

    void temp_mask(matrix_folder* theta);
	

protected:

	//cuda fft plans
	cufftHandle c2r_plan;
	cufftHandle r2c_plan;
	bool tempadded;
    int* mask_tempadded;
    matrix_device_real* theta_tempadded;


public:
    ~masktemp();

	//init the operator
    static masktemp* init_operator(std::vector<int> dimension);

    void temp_circle(matrix_folder* theta, CUDA_FLOAT_REAL radius);
    void temp_rectangle(matrix_folder* theta, CUDA_FLOAT_REAL width);
    
    // calculate frozen operator
    void calculate_operator(matrix_folder* theta);

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















