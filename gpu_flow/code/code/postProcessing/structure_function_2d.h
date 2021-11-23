#ifndef STRUCTURE_FUNCTION_2D_H
#define STRUCTURE_FUNCTION_2D_H

#include <vector>
#include <cufft.h>

#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"

class structure_function_2d {

public:
	structure_function_2d(std::vector<int> dimensions, std::vector<CUDA_FLOAT_REAL> cube_length);
	~structure_function_2d();

	
	/**
	* calculates the structure function of the temperature field at z = 0 (in the vertical center of the fluid layer)
	* @param min_r_radius is the minimal radius
 	* @param max_r_radius is the maximal radius 
	* @param number_of_samples defines the number of sampling points between min_r_radius and max_r_radius
	* @return returns a matrix_folder containing the structure function of the temperature field
	*/
	matrix_folder* calculate_temperature_structure_function_x(matrix_folder* theta, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples);

	/**
	* calculates the structure function of the temperature field at z = 0 (in the vertical center of the fluid layer)
	* @param min_r_radius is the minimal radius
 	* @param max_r_radius is the maximal radius 
	* @param number_of_samples defines the number of sampling points between min_r_radius and max_r_radius
	* @return returns a matrix_folder containing the structure function of the temperature field
	*/
	matrix_folder* calculate_temperature_structure_function_y(matrix_folder* theta, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples);

	/**
	* calculates the structure function of u_1 in x directions
	*/
	matrix_folder* calculate_u_1_structure_function_x(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples);

	/**
	* calculates the structure function of u_1 in y directions
	*/
	matrix_folder* calculate_u_1_structure_function_y(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples);

	/**
	* calculates the structure function of u_1 in x directions
	*/
	matrix_folder* calculate_u_2_structure_function_x(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples);

	/**
	* calculates the structure function of u_1 in y directions
	*/
	matrix_folder* calculate_u_2_structure_function_y(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples);

protected:
	int dim_x;
	int dim_y;
	int dim_z;

	CUDA_FLOAT_REAL cube_length_x;
	CUDA_FLOAT_REAL cube_length_y;
	CUDA_FLOAT_REAL cube_length_z;

	//cuda fft plans
	//cufftHandle c2r_plan;
	//cufftHandle r2c_plan;

private:
	
	void calculate_structure_function(matrix_folder* theta, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, int order, CUDA_FLOAT* output);

	void calculate_structure_function_u_1(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, int order, CUDA_FLOAT* output);

	void calculate_structure_function_u_2(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, int order, CUDA_FLOAT* output);

	void reduce(CUDA_FLOAT* input_used_as_memory, int number_of_entries, CUDA_FLOAT* output);
};

__global__ void reduce_step(CUDA_FLOAT* input_used_as_memory, int size, CUDA_FLOAT* output);

//builds abs of input_and_output and raises it to the power of exponent
__global__ void abs_and_square(CUDA_FLOAT_REAL* input_and_output, int number_of_real_matrix_elements, CUDA_FLOAT_REAL exponent);

__global__ void mult_conj_pointwise_on_device(CUDA_FLOAT* input,CUDA_FLOAT* to_mult , CUDA_FLOAT* result, int number_of_matrix_entries);

//twiddels the input array in the right way to get the Fourier coefficients of T(x+r_1, y+r_2, z=0) - T(x, y, z=0)
__global__ void twiddle_by_shift_temperature(CUDA_FLOAT* input, int columns, int rows, int matrices, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, CUDA_FLOAT* output, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z);

//twiddels the input array in the right way to get the Fourier coefficients of u_1(x+r_1, y+r_2, z=0) - u_1(x, y, z=0)
__global__ void twiddle_by_shift_u_1(CUDA_FLOAT* f_input, CUDA_FLOAT* g_input, int columns, int rows, int matrices, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, CUDA_FLOAT* output, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z);

//twiddels the input array in the right way to get the Fourier coefficients of u_2(x+r_1, y+r_2, z=0) - u_2(x, y, z=0)
__global__ void twiddle_by_shift_u_2(CUDA_FLOAT* f_input, CUDA_FLOAT* g_input, int columns, int rows, int matrices, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, CUDA_FLOAT* output, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z);


//some simple device functions
__device__ int get_global_index_struct_function();
__device__ int get_current_matrix_index_struct_function(int total_index, int logic_index, int columns, int rows, int matrices);
__device__ CUDA_FLOAT_REAL get_wave_number_by_index_real_matrix_structure_function(int logic_index, int col, int ro, int mat, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z);

//to create configurations for global calls
__host__ dim3 create_block_dimension_real_matrix_structure_function(int number_of_matrix_entries);
__host__ dim3 create_grid_dimension_real_matrix_structure_function(int number_of_matrix_entries);

__global__ static void scale_on_device(CUDA_FLOAT* input, CUDA_FLOAT_REAL factor, CUDA_FLOAT* result, int number_of_matrix_entries);
#endif


