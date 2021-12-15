#ifndef ENERGY_SPECTRUM_2D_H
#define ENERGY_SPECTRUM_2D_H

#include <vector>
#include <cufft.h>

#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"

class energy_spectrum_2d {

public:
	energy_spectrum_2d(std::vector<int> dimensions, std::vector<CUDA_FLOAT_REAL> cube_length);
	~energy_spectrum_2d();

	matrix_folder* calculate_energy_spectrum(matrix_folder* velocity_at_z_zero);

protected:
	int dim_x;
	int dim_y;
	int dim_z;

	CUDA_FLOAT_REAL cube_length_x;
	CUDA_FLOAT_REAL cube_length_y;
	CUDA_FLOAT_REAL cube_length_z;

private:


};


__global__ void reduce_spectrum(CUDA_FLOAT* fft_output,int num_x_fourier, int num_y_fourier,CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z, CUDA_FLOAT* spectrum_output, int energy_spectrum_size );

__global__ void copy_complex_data_to_real_data(CUDA_FLOAT* input, int number_of_elements , CUDA_FLOAT_REAL* output );

__device__ int get_global_index_energy_spectrum();
__device__ int get_current_matrix_index_energy_spectrum(int total_index, int logic_index, int columns, int rows, int matrices);

__host__ dim3 create_block_dimension_real_matrix_energy_spectrum(int number_of_matrix_entries);
__host__ dim3 create_grid_dimension_real_matrix_energy_spectrum(int number_of_matrix_entries);

__global__ void scale_on_device_complex_spectrum(CUDA_FLOAT* input,CUDA_FLOAT_REAL factor , CUDA_FLOAT* result, int number_of_matrix_entries);

#endif






