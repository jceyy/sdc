#ifndef JELLIJELL
#define JELLIJELL
//headers needed for fft : copied from operator->freeze
//system includes

#include <vector>
#include <cuda.h>
#include <cufft.h>

//simons code
#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"
#include "./freeze_operator.h"

class masking {
	private:
		//cuda fft plans
		cufftHandle c2r_plan;
		cufftHandle r2c_plan;

		//where ze mask works or mask itself
		int* ze_mask;

		//mask_data
		matrix_device_real* theta_data;

		//private constructor accessed by init_the_mask
		masking(std::vector<int> dimension);

		//position of center of circular mask (
		CUDA_FLOAT_REAL c_x, c_y, radius, width;		//width is related to rectangle stuff we are not using atm

		//factor used in julie_do_the_thing
		CUDA_FLOAT_REAL phector;
	public:
		//constructor
		static masking* init_the_mask(std::vector<int> dimension);

		~masking();

		//main method
		void masker(matrix_folder* theta);
		
		void set_mask_parameters(CUDA_FLOAT_REAL x, CUDA_FLOAT_REAL y, CUDA_FLOAT_REAL r, CUDA_FLOAT_REAL factor);
};
//method that actually applies the mask atm
__global__ static void julie_do_the_thing(CUDA_FLOAT_REAL* mask_data, int* mask, int num_xy, int num_entries, CUDA_FLOAT_REAL factor, CUDA_FLOAT_REAL phector);
//__global__ static void apply_mask(CUDA_FLOAT_REAL* input_output, CUDA_FLOAT_REAL* mask_data, int* mask, int num_xy, int num_entries);

#endif

__host__ static dim3 create_grid_dim(int num);
__host__ static dim3 create_block_dim(int num);

//helper functions
__device__ static int get_global_index();

//to create column, row and matrix index from a global index
__device__ static void get_current_matrix_indices(int &current_col, int &current_row, int &current_matrix, int total_index, int columns, int rows, int matrices);


__global__ static void init_mask_physical_space_circle(int* mask, int columns, int rows, CUDA_FLOAT_REAL radius, CUDA_FLOAT_REAL c_x, CUDA_FLOAT_REAL c_y);
__global__ static void init_mask_physical_space_rectangle(int* mask, int columns, int rows, CUDA_FLOAT_REAL width);

