#ifndef LINEAR_IMPLICIT_OPERATOR_H
#define LINEAR_IMPLICIT_OPERATOR_H

#include <vector>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>


#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"
#include "../util/coeff.h"

#define FAST_MOZ2 (true)

class linear_implicit_operator {


private:
    linear_implicit_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL rayleigh_number, CUDA_FLOAT_REAL prandtl_number, Coeff<double>& p_coeff, CUDA_FLOAT_REAL delta_t);
	
    template<typename T>
    T pow2(T x) {
        return x*x;
    }
    template<typename T>
    T pow4(T x) {
        T x2 = x*x;
        return x2*x2;
    }
    template<typename T>
    double delta(T x, T y) {
        return (x==y)?1.:0.;
    }

protected:

	//add some arrays for coefficients on GPU
    matrix_device_real* row_theta_col_theta_device;
    matrix_device_real* row_theta_col_f_device;
    matrix_device_real* row_f_col_theta_device;
    matrix_device_real* row_f_col_f_device;


	CUDA_FLOAT_REAL cube_length_x;
	CUDA_FLOAT_REAL cube_length_y;
	CUDA_FLOAT_REAL cube_length_z;

	CUDA_FLOAT_REAL prandtlNumber;
	CUDA_FLOAT_REAL deltaT;

    // The projection coefficients
    Coeff<double>& coeff;

public:
	~linear_implicit_operator();

	//init the operator
    static linear_implicit_operator* init_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL rayleigh_number, CUDA_FLOAT_REAL prandtl_number, Coeff<double>& p_coeff, CUDA_FLOAT_REAL delta_t);

	/*!
	* calculates the linear operator L(\theta, f, g, F, G)
	* @return returns an array of size 5 of type matrix_folder* which holds the results of the operator for \theta, f, g, F, G
	*/
	matrix_folder** calculate_operator(matrix_folder* theta, matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G);

};

__device__ static int get_global_index();
__device__ static void get_current_matrix_indices(int &current_col, int &current_row, int &current_matrix, int total_index, int columns, int rows, int matrices);

__global__ static void calculate_theta_f_implicit(CUDA_FLOAT* theta_data, CUDA_FLOAT* f_data, CUDA_FLOAT* theta_matrix_out_data, CUDA_FLOAT* f_matrix_out_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL* row_theta_col_theta, CUDA_FLOAT_REAL* row_theta_col_f, CUDA_FLOAT_REAL* row_f_col_theta, CUDA_FLOAT_REAL* row_f_col_f);

__global__ static void calculate_theta_f_implicit_moz2(CUDA_FLOAT* theta_data, CUDA_FLOAT* f_data, CUDA_FLOAT* theta_matrix_out_data, CUDA_FLOAT* f_matrix_out_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL* row_theta_col_theta, CUDA_FLOAT_REAL* row_theta_col_f, CUDA_FLOAT_REAL* row_f_col_theta, CUDA_FLOAT_REAL* row_f_col_f);

__global__ static void calculate_g_implicit(CUDA_FLOAT* g_data, CUDA_FLOAT* g_matrix_out_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL prandtl_number, CUDA_FLOAT_REAL delta_t);

__global__ static void calculate_F_G_implicit(CUDA_FLOAT* F_data, CUDA_FLOAT* G_data, CUDA_FLOAT* F_matrix_out_data, CUDA_FLOAT* G_matrix_out_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL prandtlNumber, CUDA_FLOAT_REAL delta_t);




#endif






















