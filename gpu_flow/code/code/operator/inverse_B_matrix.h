#ifndef INVERSE_B_MATRIX_H
#define INVERSE_B_MATRIX_H

#include <vector>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"
#include "../util/coeff.h"

#define FAST_MOZ2 (true)

class inverse_B_matrix {

public:
	matrix_folder** calculate_inverse(matrix_folder* theta_linear_output, matrix_folder* f_linear_output, matrix_folder* g_linear_output, matrix_folder* F_linear_output, matrix_folder* G_linear_output);

    static inverse_B_matrix* init_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL prandtl_number, Coeff<double>& coeff);

private:
    inverse_B_matrix(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL prandlt_number, Coeff<double> &coeff);
    template<typename T>
    T pow2(T x) {
        return x*x;
    }
    template<typename T>
    double delta(T x, T y) {
        return (x==y)?1.:0.;
    }

	//cube dimensions
	CUDA_FLOAT_REAL cube_length_x;
	CUDA_FLOAT_REAL cube_length_y;
	CUDA_FLOAT_REAL cube_length_z;
	
	//flow parameter
	CUDA_FLOAT_REAL prandtlNumber;

    // inverse B operator
    matrix_device_real* row_f_col_f_device;

};

__device__ static int get_global_index();
__device__ static void get_current_matrix_indices(int &current_col, int &current_row, int &current_matrix, int total_index, int columns, int rows, int matrices);

__global__ static void apply_B_to_theta(CUDA_FLOAT* theta_linear_data, int columns, int rows, int matrices, CUDA_FLOAT* output);
__global__ static void apply_B_to_f(CUDA_FLOAT* f_linear_data, int columns, int rows, int matrices, CUDA_FLOAT* output, CUDA_FLOAT_REAL* row_f_col_f);
__global__ static void apply_B_to_f_moz2(CUDA_FLOAT* f_linear_data, int columns, int rows, int matrices, CUDA_FLOAT *output, CUDA_FLOAT_REAL *row_f_col_f);
__global__ static void apply_B_to_g(CUDA_FLOAT* g_linear_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL prandtl_number, CUDA_FLOAT* output);
__global__ static void apply_B_to_F_and_G(CUDA_FLOAT* F_linear_data, CUDA_FLOAT* G_linear_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL prandtl_number, CUDA_FLOAT* output_F, CUDA_FLOAT* output_G);

#endif






















