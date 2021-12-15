#ifndef IMPLICIT_EXPLICIT_TIMESTEP_H
#define IMPLICIT_EXPLICIT_TIMESTEP_H

//cuda includes
#include <cuda.h>

//my includes 
// TODO rm util
#include "../util/util.h"
#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"
#include "../output/matrix_folder_writer.h"
#include "../operator/linear_implicit_operator.h"
#include "../operator/inverse_B_matrix.h"
#include "../operator/nonlinear_operator_rayleigh_noslip.h"

class implicit_explicit_step {

private:
	implicit_explicit_step(linear_implicit_operator* impl_operator, inverse_B_matrix* inv_B, nonlinear_operator_rayleigh_noslip* nonlin_op, CUDA_FLOAT_REAL delta_t);
	
	
	//operators
	nonlinear_operator_rayleigh_noslip* nonlinear_operator;
	linear_implicit_operator* implicit_op;
	inverse_B_matrix* inverse_B_op;

protected:

    matrix_folder** res_1;

	void copy_data_to_folder(matrix_folder* src, matrix_folder* target);

	void dealias_data(matrix_folder** data_vector);

public:
	~implicit_explicit_step();

	//this method should be virtual and overwritten; but CUDA doesnt support this yet
	static implicit_explicit_step* init_timestepping(linear_implicit_operator* impl_operator, inverse_B_matrix* inv_B, nonlinear_operator_rayleigh_noslip* nonlin_op, CUDA_FLOAT_REAL delta_t);

	//this method should be virtual
	void step_time(matrix_folder** input_and_output, CUDA_FLOAT_REAL delta_t);

};

__device__ static int get_global_index();
//to create column, row and matrix index from a global index
__device__ static void get_current_matrix_indices(int &current_col, int &current_row, int &current_matrix,
                                                  int total_index, int columns, int rows, int matrices);

__global__ static void dealias_implicit(CUDA_FLOAT* data, int columns, int rows, int matrices);
__global__ static void copy_data(CUDA_FLOAT* src_data, CUDA_FLOAT* target_data, int columns, int rows, int matrices);


#endif 

