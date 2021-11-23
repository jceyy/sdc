#include "inverse_B_matrix.h"

matrix_folder** inverse_B_matrix::calculate_inverse(matrix_folder* theta_linear_output, matrix_folder* f_linear_output, matrix_folder* g_linear_output, matrix_folder* F_linear_output, matrix_folder* G_linear_output){

	#ifdef DEBUG
		std::cout << "calculate inverse B operator" << std::endl;
	#endif

	matrix_folder** return_folder = new matrix_folder*[5];


	//calculations for theta
    matrix_device* theta_result = new matrix_device(theta_linear_output->get_matrix(0), matrix_device::noInit);
	CUDA_FLOAT* theta_linear_data = theta_linear_output->get_matrix(0)->get_data();
	std::vector<int> theta_dim = theta_linear_output->get_matrix(0)->get_matrix_dimension();
    apply_B_to_theta<<<theta_result->create_grid_dimension(),theta_result->create_block_dimension()>>> (theta_linear_data, theta_dim.at(0), theta_dim.at(1), theta_dim.at(2), theta_result->get_data());

	//calculations for f
    matrix_device* f_result = new matrix_device(f_linear_output->get_matrix(0), matrix_device::noInit);
	CUDA_FLOAT* f_linear_data = f_linear_output->get_matrix(0)->get_data();
    std::vector<int> f_dim = f_linear_output->get_matrix(0)->get_matrix_dimension();
    if(f_dim.at(2)==2 && FAST_MOZ2) {
        apply_B_to_f_moz2<<<f_result->create_grid_dimension(),f_result->create_block_dimension()>>> (f_linear_data, f_dim.at(0), f_dim.at(1), f_dim.at(2), f_result->get_data(), row_f_col_f_device->get_data());
    } else {
        apply_B_to_f<<<f_result->create_grid_dimension(),f_result->create_block_dimension()>>> (f_linear_data, f_dim.at(0), f_dim.at(1), f_dim.at(2), f_result->get_data(), row_f_col_f_device->get_data());
    }


	//calculations for g
    matrix_device* g_result = new matrix_device(g_linear_output->get_matrix(0), matrix_device::noInit);
	CUDA_FLOAT* g_linear_data = g_linear_output->get_matrix(0)->get_data();
	std::vector<int> g_dim = g_linear_output->get_matrix(0)->get_matrix_dimension();
    apply_B_to_g<<<g_result->create_grid_dimension(),g_result->create_block_dimension()>>> (g_linear_data, g_dim.at(0), g_dim.at(1), g_dim.at(2), cube_length_x, cube_length_y, prandtlNumber, g_result->get_data());


    // calculations for F and G
    matrix_device* F_result = new matrix_device(F_linear_output->get_matrix(0), matrix_device::noInit);
    matrix_device* G_result = new matrix_device(G_linear_output->get_matrix(0), matrix_device::noInit);
    CUDA_FLOAT* F_linear_data = F_linear_output->get_matrix(0)->get_data();
    CUDA_FLOAT* G_linear_data = G_linear_output->get_matrix(0)->get_data();
    std::vector<int> F_dim = F_linear_output->get_matrix(0)->get_matrix_dimension();
    apply_B_to_F_and_G<<<F_result->create_grid_dimension(),F_result->create_block_dimension()>>> (F_linear_data, G_linear_data, F_dim.at(0), F_dim.at(1), F_dim.at(2), prandtlNumber, F_result->get_data(), G_result->get_data());

	//set return values
	matrix_folder* theta_return_folder = new matrix_folder(1);
	matrix_folder* f_return_folder = new matrix_folder(1);
	matrix_folder* g_return_folder = new matrix_folder(1);
	matrix_folder* F_return_folder = new matrix_folder(1);
	matrix_folder* G_return_folder = new matrix_folder(1);
	theta_return_folder->add_matrix(0, theta_result);
	f_return_folder->add_matrix(0, f_result);
	g_return_folder->add_matrix(0, g_result);
	F_return_folder->add_matrix(0, F_result);
	G_return_folder->add_matrix(0, G_result);
	return_folder[0] = theta_return_folder;
	return_folder[1] = f_return_folder;
	return_folder[2] = g_return_folder;
	return_folder[3] = F_return_folder;
    return_folder[4] = G_return_folder;

	return return_folder;
}


inverse_B_matrix* inverse_B_matrix::init_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL prandtl_number, Coeff<double>& coeff){
    inverse_B_matrix* op = new inverse_B_matrix(dimension, cube_length, prandtl_number, coeff);
	return op;
}


inverse_B_matrix::inverse_B_matrix(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL prandtl_number, Coeff<double>& coeff) {

    /*if(dimension.at(2) != 2) {
        EXIT_ERROR("inverse_B_matrix is currently implemented only for 2 vertical ansatz functions!");
    }*/

	//cube length
    cube_length_x = cube_length.at(0);
    cube_length_y = cube_length.at(1);
    cube_length_z = cube_length.at(2);

    prandtlNumber = prandtl_number;

    // Code to create inverse B operator for f (which is nondiagonal)
    const int num_x = dimension.at(0);
    const int num_y = dimension.at(1);
    const int num_z = dimension.at(2);

    // store inverse operator
    std::vector<int> store_dim(3);
    store_dim.at(0) = num_x;
    store_dim.at(1) = num_y;
    store_dim.at(2) = num_z * num_z;
    // Special compression for diagonal moz=2 case
    if(num_z==2 && FAST_MOZ2) store_dim.at(2) = 2;

    matrix_host_real* row_f_col_f = new matrix_host_real(store_dim);
    row_f_col_f->init_zeros();

    gsl_matrix* inv_B = gsl_matrix_alloc(num_z, num_z);
    gsl_matrix* tmp_B = gsl_matrix_alloc(num_z, num_z);
    gsl_permutation* p = gsl_permutation_alloc(num_z);

    // loop over columns
    for(int k = 0; k < num_x; k++) {
        //mode in x direction
        CUDA_FLOAT_REAL mode_x = k*2.0*M_PI / cube_length_x;

        //loop over rows
        for(int l = 0; l < num_y; l++) {

            //mode in y direction
            CUDA_FLOAT_REAL mode_y = 2.0*M_PI / cube_length_y;
            if(l < (num_y/2)+1) mode_y *= l;
            else mode_y *= (l - num_y);

            //.... square of absolute value of mode
            CUDA_FLOAT_REAL mode_sq = mode_x*mode_x + mode_y*mode_y;

            // For the (0,0)-mode, f is projected to 0 because it has zero mean value.
            if(k == 0 && l == 0) {
                gsl_matrix_set_zero(inv_B);
            } else {
                if(num_z==2 && FAST_MOZ2) {
                    // for 2 modes, B is diagonal and can be inverted manually
                    gsl_matrix_set_zero(inv_B);
                    gsl_matrix_set(inv_B, 0, 0, prandtl_number / ( mode_sq*(mode_sq - coeff.Iac(0,0)) ) );
                    gsl_matrix_set(inv_B, 1, 1, prandtl_number / ( mode_sq*(mode_sq - coeff.Iac(1,1)) ) );
                } else {
                    // for higher modes, B is inverted by LU-decomposition
                    gsl_matrix_set_zero(tmp_B);
                    for(int m = 0; m < num_z; ++m) {
                        gsl_matrix_set(tmp_B, m, m, 0.5);
                        for(int m1 = 0; m1 < num_z; ++m1) {
                            gsl_matrix_set(tmp_B, m, m1, mode_sq*(mode_sq*delta(m,m1) - coeff.Iac(m,m1)) / prandtl_number);
                        }
                    }
                    int signum = 0;
                    gsl_linalg_LU_decomp(tmp_B, p, &signum);
                    gsl_linalg_LU_invert(tmp_B, p, inv_B);

                }
            }

            // store coefficients from inverse
            std::vector<int> entry_index(3);
            entry_index[0] = k; entry_index[1] = l;
            if(num_z==2 && FAST_MOZ2) {
                // special compression for diagonal moz=2 case
                for(int m = 0; m < num_z; ++m) {
                    entry_index[2] = m;
                    row_f_col_f->set_entry(entry_index, gsl_matrix_get(inv_B, m, m));
                }
            } else {
                for(int m = 0; m < num_z; ++m) {
                    for(int m1 = 0; m1 < num_z; ++m1) {
                        entry_index[2] = m*num_z+m1;
                        row_f_col_f->set_entry(entry_index, gsl_matrix_get(inv_B, m, m1));
                    }
                }
            }
        }
    }
    gsl_matrix_free(inv_B);
    gsl_matrix_free(tmp_B);
    gsl_permutation_free(p);

    // move data to device
    row_f_col_f_device = new matrix_device_real(row_f_col_f, matrix_device_real::Copy);

    //free memory
    delete row_f_col_f;
}


__device__ static int get_global_index() {
    return (threadIdx.x + (threadIdx.y + (threadIdx.z + (blockIdx.x + (blockIdx.y + (blockIdx.z)
            * gridDim.y) * gridDim.x) * blockDim.z) * blockDim.y) * blockDim.x);
}

__device__ static void get_current_matrix_indices(int& current_col, int& current_row, int& current_matrix,
                                                  int total_index, int columns, int rows, int matrices) {

    int xysize = rows * columns;

    current_col = (total_index % columns);
    current_row = ((total_index % xysize) / columns);
    current_matrix = ((total_index % (xysize * matrices)) / xysize);
}

__device__ static CUDA_FLOAT_REAL get_wave_number_x(int col, int columns, CUDA_FLOAT_REAL cube_length_x){
    return (M_PI + M_PI) * (col / cube_length_x);
}
__device__ static CUDA_FLOAT_REAL get_wave_number_y(int row, int rows, CUDA_FLOAT_REAL cube_length_y){
    if(row >= (rows/2)+1) row -= rows;
    return (M_PI + M_PI) * (row / cube_length_y);
}

__global__ void apply_B_to_theta(CUDA_FLOAT* theta_linear_data, int columns, int rows, int matrices, CUDA_FLOAT* output) {
    int total_index = get_global_index();

	//apply inverse of 0.5 to all entries
	if(total_index < columns*rows*matrices) {
        // load
        CUDA_FLOAT val = theta_linear_data[total_index];

        // calc
        val.x *= 2.0;
        val.y *= 2.0;

        // store
        output[total_index] = val;
	}

}


// Calculate the inverse timestep of f and theta by
// applying the inverse timestepping matrix row_f_col_f (of size (moz^2) for each k,l)
// to the source field and adding the values up in the dest field.
__global__ void apply_B_to_f(CUDA_FLOAT* f_linear_data, int columns, int rows, int matrices, CUDA_FLOAT *output, CUDA_FLOAT_REAL *row_f_col_f){
    int out_index = get_global_index();
    if(out_index < columns*rows*matrices) {

        int current_col = 0, current_row = 0, current_mat = 0;
        get_current_matrix_indices(current_col, current_row, current_mat, out_index, columns, rows, matrices);

        CUDA_FLOAT out_f;
        out_f.x = 0.;
        out_f.y = 0.;

        for(int m1 = 0; m1 < matrices; ++m1) {
            int in_index = current_col + columns * current_row + (columns*rows)*m1;
            int op_index = current_col + columns * current_row + (columns*rows)*(current_mat*matrices+m1);

            // load
            CUDA_FLOAT in_f = f_linear_data[in_index];
            CUDA_FLOAT_REAL op_f_f = row_f_col_f[op_index];

            // calc
            out_f.x += op_f_f * in_f.x;
            out_f.y += op_f_f * in_f.y;
        }

        // store
        output[out_index] = out_f;
    }
}


// Shorter form of apply_B_to_f. For moz=2, the timestepping operator is
// diagonal, so that only diag(op) is stored in the fields (see special code in constructor)
// and the operator can be applied without an extra inner loop.
__global__ void apply_B_to_f_moz2(CUDA_FLOAT* f_linear_data, int columns, int rows, int matrices, CUDA_FLOAT *output, CUDA_FLOAT_REAL *row_f_col_f) {
    int total_index = get_global_index();
    if(total_index < columns*rows*matrices) {
        // load
        CUDA_FLOAT in_f = f_linear_data[total_index];
        CUDA_FLOAT_REAL op_f_f = row_f_col_f[total_index];

        // calc
        CUDA_FLOAT out_f;
        out_f.x = op_f_f * in_f.x;
        out_f.y = op_f_f * in_f.y;

        // store
        output[total_index] = out_f;
    }

}


__global__ void apply_B_to_g(CUDA_FLOAT* g_linear_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL prandtl_number, CUDA_FLOAT* output) {
    int total_index = get_global_index();
    int current_col = 0, current_row = 0, vertical_function_number = 0;
    get_current_matrix_indices(current_col, current_row, vertical_function_number, total_index, columns, rows, matrices);

    CUDA_FLOAT_REAL mode_x = get_wave_number_x(current_col, columns, cube_length_x);
    CUDA_FLOAT_REAL mode_y = get_wave_number_y(current_row, rows, cube_length_y);
    CUDA_FLOAT_REAL mode_sq = mode_x*mode_x + mode_y*mode_y;

    CUDA_FLOAT val;

    if(total_index < columns*rows*matrices) {
        //apply inverse to all entries
        if(mode_sq != 0.0) {
            // (B)^(-1) = (-mode_abs / (2 Pr))^(-1)
            CUDA_FLOAT_REAL factor = (-2.0) * prandtl_number / mode_sq;

            val = g_linear_data[total_index];
            val.x *= factor;
            val.y *= factor;
        } else {
            //set static average to zero, this is handled by F
            val.x = 0.0;
            val.y = 0.0;
        }

        // store
        output[total_index] = val;
    }
}


__global__ void apply_B_to_F_and_G(CUDA_FLOAT* F_linear_data, CUDA_FLOAT* G_linear_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL prandtl_number, CUDA_FLOAT* output_F, CUDA_FLOAT* output_G) {
    int total_index = get_global_index();

	//apply inverse of 0.5 to all entries
    if(total_index < columns*rows*matrices) {

        // load
        CUDA_FLOAT in_F = F_linear_data[total_index];
        CUDA_FLOAT in_G = G_linear_data[total_index];

        // calc
        CUDA_FLOAT out_F, out_G;
        CUDA_FLOAT_REAL factor = 2.0 * prandtl_number;
        out_F.x = factor * in_F.x;
        out_F.y = 0.0;
        out_G.x = factor * in_G.x;
        out_G.y = 0.0;

        // store
        output_F[total_index] = out_F;
        output_G[total_index] = out_G;
	}

}
















