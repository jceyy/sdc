#ifndef CALCULATE_TEMPERATURE_OPERATOR_H
#define CALCULATE_TEMPERATURE_OPERATOR_H

#include <vector>
#include <cuda.h>
#include <cufft.h>

#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"


class calculate_temperature_operator {

public:
	static calculate_temperature_operator* init(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length);

    matrix_folder_real* calculate_operator(matrix_folder* theta, const int num_z = 5);
    matrix_folder_real* calculate_operator_at(matrix_folder* theta, const vector<CUDA_FLOAT_REAL>& at_z);

	~calculate_temperature_operator();

protected:
	CUDA_FLOAT_REAL cube_length_x;
	CUDA_FLOAT_REAL cube_length_y;
	CUDA_FLOAT_REAL cube_length_z;
	
	//cuda fft plans
	cufftHandle c2r_plan;

private:
	calculate_temperature_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length);


};

__host__ static dim3 create_grid_dim(int number_of_matrix_entries);
__host__ static dim3 create_block_dim(int number_of_matrix_entries);

__device__ static int get_global_index();

__global__ static void mult_add_pointwise(CUDA_FLOAT_REAL* input_1, CUDA_FLOAT_REAL factor_1, int number_of_matrix_entries, CUDA_FLOAT_REAL* input_and_output);


#endif



