#ifndef NONLINEAR_OPERATOR_RAYLEIGH_NOSLIP_H
#define NONLINEAR_OPERATOR_RAYLEIGH_NOSLIP_H

//system includes
#include <vector>
#include <cuda.h>
#include <cufft.h>

//my defines
#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"
#include "../util/coeff.h"
// TODO rm util
#include "../util/util.h"


class nonlinear_operator_rayleigh_noslip {


private:
    nonlinear_operator_rayleigh_noslip(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL prandtl_Number, Coeff<double>& p_coeff, CUDA_FLOAT_REAL penalization_eta = 0);
	

protected:
	//dimension of our cube
	CUDA_FLOAT_REAL cube_length_x;
	CUDA_FLOAT_REAL cube_length_y;
	CUDA_FLOAT_REAL cube_length_z;

	//cuda fft plans
	cufftHandle c2r_plan;
	cufftHandle r2c_plan;

    // The projection coefficients
    Coeff<double>& coeff;

	//coefficients
	CUDA_FLOAT_REAL prandtl_number;
#ifdef PENALIZED
	CUDA_FLOAT_REAL eta;

	//penalizations mask on CUDA device
    CUDA_FLOAT_REAL* mask;
#endif /* PENALIZED */

public:
    ~nonlinear_operator_rayleigh_noslip();

	//init the operator
    static nonlinear_operator_rayleigh_noslip* init_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL prandtl_Number, Coeff<double>& p_coeff, CUDA_FLOAT_REAL penalization_eta = 0);

	/*!
	* calculates the linear operator L(\theta, f, g, F, G)
	* @return returns an array of size 5 of type matrix_folder* which holds the results of the operator for \theta, f, g, F, G
	*/
	matrix_folder** calculate_operator(matrix_folder* theta, matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G);

    // TODO REMOVE
    /*void tmp_print(CUDA_FLOAT_REAL* data_d, CUDA_FLOAT_REAL factor, int mox, int moy, int moz, const char* lbl) {
        CUDA_FLOAT_REAL* data_h = new CUDA_FLOAT_REAL[mox * moy * moz];
        cudaMemcpy(data_h, data_d, sizeof(CUDA_FLOAT_REAL) * mox * moy * moz, cudaMemcpyDeviceToHost);
        cout << lbl << endl;
        for(int j = 0; j < moy; ++j) {
            for(int i = 0; i < mox; ++i) {
                cout << i << " " << j << " ";
                for(int k = 0; k < moz; ++k) {
                    cout << "(" << factor * data_h[i + j * mox + k * mox * moy] << ") ";
                }
                cout << endl;
            }
        }
        delete [] data_h;
    }
    void tmp_print(CUDA_FLOAT* data_d, CUDA_FLOAT_REAL factor, int mox, int moy, int moz, const char* lbl, bool mep = false) {
        CUDA_FLOAT* data_h = new CUDA_FLOAT[mox * moy * moz];
        cudaMemcpy(data_h, data_d, sizeof(CUDA_FLOAT) * mox * moy * moz, cudaMemcpyDeviceToHost);
        cout << lbl << endl;
        if(mep) {
            cout << "ATTENTION: Matrix is manipulated" << endl;
        }
        for(int j = 0; j < moy; ++j) {
            for(int i = 0; i < mox; ++i) {
                cout << i << " " << j << " ";
                for(int k = 0; k < moz; ++k) {
                    if(mep && i==0 && j == 0) data_h[i + j * mox + k * mox * moy].x = (1);
                    cout << "(" << factor * data_h[i + j * mox + k * mox * moy].x << ","
                             << factor * data_h[i + j * mox + k * mox * moy].y << ") ";
                }
                cout << endl;
            }
        }
        cudaMemcpy(data_d, data_h, sizeof(CUDA_FLOAT) * mox * moy * moz, cudaMemcpyHostToDevice);
        delete [] data_h;
    }*/

};

__host__ static dim3 create_grid_dim(int num);
__host__ static dim3 create_block_dim(int num);

//helper functions
__device__ static int get_global_index();
//to create column, row and matrix index from a global index
__device__ static void get_current_matrix_indices(int &current_col, int &current_row, int &current_matrix, int total_index, int columns, int rows, int matrices);

// Create derivatives of fourier-space matrices
__global__ static void create_derivative_x(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x);
__global__ static void create_derivative_y(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_y);
__global__ static void create_second_derivative_xx(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x);
__global__ static void create_second_derivative_yy(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_y);
__global__ static void create_second_derivative_xy(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y);
__global__ static void create_laplace_xy(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y);


__global__ static void mult_pointwise_real_matrix(CUDA_FLOAT_REAL* input_1, CUDA_FLOAT_REAL* input_2, CUDA_FLOAT_REAL* output, CUDA_FLOAT_REAL factor, int num_entries);
__global__ static void add_dc_part(CUDA_FLOAT* g, CUDA_FLOAT* F, int columns, int rows, int matrices);
__global__ static void addup_only_dc_part(CUDA_FLOAT* F, CUDA_FLOAT* f, int f_columns, int f_rows, int f_matrices, CUDA_FLOAT_REAL factor);
__global__ static void scale_on_device(CUDA_FLOAT* input, CUDA_FLOAT_REAL factor, CUDA_FLOAT* result, int number_of_matrix_entries);
__global__ static void add_mult_pointwise_on_device(CUDA_FLOAT* input, CUDA_FLOAT* to_add, CUDA_FLOAT_REAL factor, CUDA_FLOAT* result, int number_of_matrix_entries);

#ifdef PENALIZED
//masks for penalizations
__global__ static void init_mask_physical_space_rectangle(CUDA_FLOAT_REAL* mask, int num_elements_real_x,int num_elements_real_y);
__global__ static void init_mask_physical_space_circle(CUDA_FLOAT_REAL* mask, int num_elements_real_x,int num_elements_real_y);
__global__ static void init_mask_physical_space_disk_array(CUDA_FLOAT_REAL* mask, int columns, int rows, int number_of_disks_x, int number_of_disks_y);
__global__ static void init_mask_physical_space_everywhere(CUDA_FLOAT_REAL* mask, int num_elements_real_x,int num_elements_real_y);
#endif /* PENALIZED */

#endif

















