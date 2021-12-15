#include "linear_implicit_operator.h"

// Note:
// <C1S1> = Ia1(0,0) =  0.6973
// <C2S2> = Ia1(1,1) = -0.6904
// <C1C1''> = Iac(0,0) = -12.30
// <C2C2''> = Iac(1,1) = -46.05
// lambda_1 = 4.73
// lambda_2 = 7.85


linear_implicit_operator::linear_implicit_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL rayleigh_number, CUDA_FLOAT_REAL prandtl_number, Coeff<double>& p_coeff, CUDA_FLOAT_REAL delta_t) : coeff(p_coeff){
	
    const int num_x = dimension.at(0);
    const int num_y = dimension.at(1);
    const int num_z = dimension.at(2);

    /*if(num_z != 2) {
        EXIT_ERROR("not able to use more than 2 vertical ansatz functions with implicit operator!");
    }*/

	//store parameters
    cube_length_x = cube_length.at(0);
    cube_length_y = cube_length.at(1);
    cube_length_z = cube_length.at(2);
    prandtlNumber = prandtl_number;
    deltaT = delta_t;

    // store inverse operator
    std::vector<int> store_dim(3);
    store_dim.at(0) = num_x;
    store_dim.at(1) = num_y;
    store_dim.at(2) = num_z * num_z;
    // Special compression for diagonal moz=2 case
    if(num_z==2 && FAST_MOZ2) store_dim.at(2) = 2;

    matrix_host_real* row_theta_col_theta = new matrix_host_real(store_dim);
	row_theta_col_theta->init_zeros();
    matrix_host_real* row_theta_col_f = new matrix_host_real(store_dim);
	row_theta_col_f->init_zeros();
    matrix_host_real* row_f_col_theta = new matrix_host_real(store_dim);
	row_f_col_theta->init_zeros();
    matrix_host_real* row_f_col_f = new matrix_host_real(store_dim);
	row_f_col_f->init_zeros();

	//loop over columns
    for(int k = 0; k < num_x; k++) {
		//mode in x direction
        CUDA_FLOAT_REAL mode_x = k*2.0*M_PI / cube_length_x;

		//loop over rows
        for(int l = 0; l < num_y; l++) {

			//mode in y direction
            CUDA_FLOAT_REAL mode_y = 2.0*M_PI / cube_length_y;
            if(l < (num_y/2)+1) mode_y *= l;
            else mode_y *= (num_y - l);

			//.... square of absolute value of mode
            CUDA_FLOAT_REAL mode_sq = mode_x*mode_x + mode_y*mode_y;
		
			//create matrices
            gsl_matrix* ident = gsl_matrix_alloc(2*num_z, 2*num_z);
            gsl_matrix* inv_B = gsl_matrix_alloc(2*num_z, 2*num_z);

            // Set identity matrix
            gsl_matrix_set_identity(ident);

            // For the (0,0)-mode, f is projected to 0 because it has zero mean value.
            if(k == 0 && l == 0) {
                // Set f-part to zero for identity matrix (!)
                for(int m = num_z; m < 2*num_z; ++m) {
                    gsl_matrix_set(ident, m, m, 0.0);
                }

                // B is diagonal and can be inverted manually
				gsl_matrix_set_zero(inv_B);
                for(int m = 0; m < num_z; ++m) {
                    gsl_matrix_set(inv_B, m, m, 2.0);
                    // upper part (for ff) is zero
                }
				
            } else {

                if(num_z==2 && FAST_MOZ2) {
                    // for 2 modes, B is diagonal and can be inverted manually
                    gsl_matrix_set_zero(inv_B);
                    gsl_matrix_set(inv_B, 0, 0, 2.0);
                    gsl_matrix_set(inv_B, 1, 1, 2.0);
                    gsl_matrix_set(inv_B, 2, 2, prandtl_number / ( mode_sq*(mode_sq - coeff.Iac(0,0)) ) );
                    gsl_matrix_set(inv_B, 3, 3, prandtl_number / ( mode_sq*(mode_sq - coeff.Iac(1,1)) ) );

                } else {
                    // for higher modes, B is inverted by LU-decomposition
                    gsl_matrix* tmp_B = gsl_matrix_alloc(2*num_z,2*num_z);
                    gsl_matrix_set_zero(tmp_B);
                    for(int m = 0; m < num_z; ++m) {
                        gsl_matrix_set(tmp_B, m, m, 0.5);
                        for(int m1 = 0; m1 < num_z; ++m1) {
                            gsl_matrix_set(tmp_B, m + num_z, m1 + num_z, mode_sq*(mode_sq*delta(m,m1) - coeff.Iac(m,m1)) / prandtl_number);
                        }
                    }
                    gsl_permutation* p = gsl_permutation_alloc(2*num_z);
                    int signum = 0;
                    gsl_linalg_LU_decomp(tmp_B, p, &signum);
                    gsl_linalg_LU_invert(tmp_B, p, inv_B);

                    gsl_matrix_free(tmp_B);
                    gsl_permutation_free(p);
                }
			}
			
            // create (explicit) L matrix
            gsl_matrix* L = gsl_matrix_alloc(2*num_z, 2*num_z);
            gsl_matrix_set_zero(L);

            // Fill elements of L matrix for T and f
            for(int m = 0; m < num_z; ++m) {
                for(int m1 = 0; m1 < num_z; ++m1) {
                    // L^{TT}_{mm'}
                    gsl_matrix_set(L, m, m1, (-0.5)*(mode_sq + pow2((m+1)*M_PI))*delta(m,m1));
                    // L^{Tf}_{mm'}
                    gsl_matrix_set(L, m, m1 + num_z, mode_sq * coeff.Ia1(m1,m));
                    // L^{fT}_{mm'}
                    gsl_matrix_set(L, m + num_z, m1, rayleigh_number * mode_sq * coeff.Ia1(m,m1));
                    // L^{ff}_{mm'}
                    gsl_matrix_set(L, m + num_z, m1 + num_z, -mode_sq * ( (pow4(coeff.lambda(m+1))+mode_sq*mode_sq)*delta(m,m1) - 2.0*mode_sq*coeff.Iac(m,m1) ) );
                }
            }

		
            //calc: C = -delta_t * inv_B * L + 0.0 * C
            gsl_matrix* C = gsl_matrix_alloc(2*num_z,2*num_z);
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -delta_t, inv_B, L, 0.0, C);
		
			//calc: C = ident + C
			gsl_matrix_add(C, ident);


			gsl_permutation* p;
            gsl_matrix* inverse = gsl_matrix_alloc(2*num_z,2*num_z);
            if(k == 0 && l == 0) {
                // Invert only upper part (TT-matrix) of C, the rest remains zero
                gsl_matrix* upper_theta = gsl_matrix_alloc(num_z,num_z);
                for(int m = 0; m < num_z; ++m) {
                    for(int m1 = 0; m1 < num_z; ++m1) {
                        gsl_matrix_set(upper_theta, m, m1, gsl_matrix_get(C, m, m1));
                    }
                }

				int signum = 0;
                p = gsl_permutation_alloc(num_z);
                gsl_linalg_LU_decomp(upper_theta, p, &signum);
                gsl_matrix* tmp_inverse = gsl_matrix_alloc(num_z,num_z);
                gsl_linalg_LU_invert(upper_theta, p, tmp_inverse);

				//init inverse with zeros
				gsl_matrix_set_zero(inverse);
				//copy to inverse
                for(int m = 0; m < num_z; ++m) {
                    for(int m1 = 0; m1 < num_z; ++m1) {
                        gsl_matrix_set(inverse, m, m1, gsl_matrix_get(tmp_inverse, m, m1));
                    }
                }

				gsl_matrix_free(tmp_inverse);
				gsl_matrix_free(upper_theta);
			} else {
				//invert C by LU decomposition
				int signum = 0;
                p = gsl_permutation_alloc(2*num_z);
                gsl_linalg_LU_decomp(C, p, &signum);
                gsl_linalg_LU_invert(C, p, inverse);
			}
			
			// store coefficients from inverse
            std::vector<int> entry_index(3);
            entry_index[0] = k; entry_index[1] = l;
            if(num_z==2 && FAST_MOZ2) {
                // special compression for diagonal moz=2 case
                for(int m = 0; m < num_z; ++m) {
                    entry_index[2] = m;
                    row_theta_col_theta->set_entry(entry_index, gsl_matrix_get(inverse, m, m));
                    row_theta_col_f->set_entry(entry_index, gsl_matrix_get(inverse, m, m+2));
                    row_f_col_theta->set_entry(entry_index, gsl_matrix_get(inverse, m+2, m));
                    row_f_col_f->set_entry(entry_index, gsl_matrix_get(inverse, m+2, m+2));
                }
            } else {
                for(int m = 0; m < num_z; ++m) {
                    for(int m1 = 0; m1 < num_z; ++m1) {
                        entry_index[2] = m*num_z+m1;
                        row_theta_col_theta->set_entry(entry_index,
                                                       gsl_matrix_get(inverse, m, m1));
                        row_theta_col_f->set_entry(entry_index,
                                                   gsl_matrix_get(inverse, m, m1+num_z));
                        row_f_col_theta->set_entry(entry_index,
                                                   gsl_matrix_get(inverse, m+num_z, m1));
                        row_f_col_f->set_entry(entry_index,
                                               gsl_matrix_get(inverse, m+num_z, m1+num_z));
                    }
                }
            }

			//free memory
			gsl_matrix_free(ident);
			gsl_matrix_free(inv_B);
			gsl_matrix_free(L);
			gsl_matrix_free(C);
			gsl_matrix_free(inverse);
			gsl_permutation_free(p);
			
		}

	}

	//transfer coefficients to GPU
    row_theta_col_theta_device = new matrix_device_real(row_theta_col_theta, matrix_device_real::Copy);
    row_theta_col_f_device = new matrix_device_real(row_theta_col_f, matrix_device_real::Copy);
    row_f_col_theta_device = new matrix_device_real(row_f_col_theta, matrix_device_real::Copy);
    row_f_col_f_device = new matrix_device_real(row_f_col_f, matrix_device_real::Copy);

	//free memory 
	delete row_theta_col_theta;
	delete row_theta_col_f;
	delete row_f_col_theta;
    delete row_f_col_f;
}
	
linear_implicit_operator::~linear_implicit_operator(){
	delete row_theta_col_theta_device;
	delete row_theta_col_f_device;
	delete row_f_col_theta_device;
	delete row_f_col_f_device;
}

linear_implicit_operator* linear_implicit_operator::init_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL rayleigh_number, CUDA_FLOAT_REAL prandtl_number, Coeff<double>& p_coeff, CUDA_FLOAT_REAL delta_t){
    linear_implicit_operator* op = new linear_implicit_operator(dimension, cube_length, rayleigh_number, prandtl_number, p_coeff, delta_t);
	return op;
}


matrix_folder** linear_implicit_operator::calculate_operator(matrix_folder* theta, matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G){

    int columns = theta->get_matrix(0)->get_matrix_size(0);
    int rows = theta->get_matrix(0)->get_matrix_size(1);
    int matrices = theta->get_matrix(0)->get_matrix_size(2);

    int F_columns = F->get_matrix(0)->get_matrix_size(0);
    int F_rows = F->get_matrix(0)->get_matrix_size(1);
    int F_matrices = F->get_matrix(0)->get_matrix_size(2);

    CUDA_FLOAT_REAL* row_theta_col_theta = row_theta_col_theta_device->get_data();
    CUDA_FLOAT_REAL* row_theta_col_f = row_theta_col_f_device->get_data();

    CUDA_FLOAT_REAL* row_f_col_theta = row_f_col_theta_device->get_data();
    CUDA_FLOAT_REAL* row_f_col_f = row_f_col_f_device->get_data();


	//input data matices
	matrix_device* theta_matrix = theta->get_matrix(0);
	CUDA_FLOAT* theta_data = theta_matrix->get_data();
	matrix_device* f_matrix = f->get_matrix(0);
	CUDA_FLOAT* f_data = f_matrix->get_data();
	matrix_device* g_matrix = g->get_matrix(0);
	CUDA_FLOAT* g_data = g_matrix->get_data();
	matrix_device* F_matrix = F->get_matrix(0);
	CUDA_FLOAT* F_data = F_matrix->get_data();
	matrix_device* G_matrix = G->get_matrix(0);
	CUDA_FLOAT* G_data = G_matrix->get_data();

	//to configure global calls
	dim3 theta_grid = theta_matrix->create_grid_dimension();
	dim3 theta_block = theta_matrix->create_block_dimension();
	dim3 F_grid = F_matrix->create_grid_dimension();
	dim3 F_block = F_matrix->create_block_dimension();

    //calc theta and f
	matrix_device* theta_matrix_out = new matrix_device(theta_matrix->get_matrix_dimension());
	CUDA_FLOAT* theta_matrix_out_data = theta_matrix_out->get_data();
    matrix_device* f_matrix_out = new matrix_device(f_matrix->get_matrix_dimension());
    CUDA_FLOAT* f_matrix_out_data = f_matrix_out->get_data();
    if(matrices == 2 && FAST_MOZ2) {
        calculate_theta_f_implicit_moz2<<<theta_grid,theta_block>>>(theta_data, f_data, theta_matrix_out_data, f_matrix_out_data, columns, rows, matrices, row_theta_col_theta, row_theta_col_f, row_f_col_theta, row_f_col_f);
    } else {
        calculate_theta_f_implicit<<<theta_grid,theta_block>>>(theta_data, f_data, theta_matrix_out_data, f_matrix_out_data, columns, rows, matrices, row_theta_col_theta, row_theta_col_f, row_f_col_theta, row_f_col_f);
    }


	//calc g
	matrix_device* g_matrix_out = new matrix_device(g_matrix->get_matrix_dimension());
	CUDA_FLOAT* g_matrix_out_data = g_matrix_out->get_data();
    calculate_g_implicit<<< theta_grid , theta_block >>>(g_data, g_matrix_out_data, columns, rows, matrices, cube_length_x, cube_length_y, prandtlNumber, deltaT);


    //calc F and G
	matrix_device* F_matrix_out = new matrix_device(F_matrix->get_matrix_dimension());
    matrix_device* G_matrix_out = new matrix_device(G_matrix->get_matrix_dimension());
    CUDA_FLOAT* F_matrix_out_data = F_matrix_out->get_data();
    CUDA_FLOAT* G_matrix_out_data = G_matrix_out->get_data();
    calculate_F_G_implicit<<<F_grid,F_block>>>(F_data, G_data, F_matrix_out_data, G_matrix_out_data, F_columns, F_rows, F_matrices, prandtlNumber, deltaT);


	//set return values
	matrix_folder** return_folder = new matrix_folder*[5];
	matrix_folder* theta_return = new matrix_folder(1);
	theta_return->add_matrix(0, theta_matrix_out);
	matrix_folder* f_return = new matrix_folder(1);
	f_return->add_matrix(0, f_matrix_out);
	matrix_folder* g_return = new matrix_folder(1);
	g_return->add_matrix(0, g_matrix_out);
	matrix_folder* F_return = new matrix_folder(1);
	F_return->add_matrix(0, F_matrix_out);
	matrix_folder* G_return = new matrix_folder(1);
	G_return->add_matrix(0, G_matrix_out);
	return_folder[0] = theta_return;
	return_folder[1] = f_return;
	return_folder[2] = g_return;
	return_folder[3] = F_return;
	return_folder[4] = G_return;

	return return_folder;
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

// Calculate the implicit operator of f and theta by
// applying the implicit matrices row_<dest>_col_<source> (of size (moz^2) for each k,l)
// to the two different source fields and adding them up in the dest field.
__global__ void calculate_theta_f_implicit(CUDA_FLOAT* theta_data, CUDA_FLOAT* f_data, CUDA_FLOAT* theta_matrix_out_data, CUDA_FLOAT* f_matrix_out_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL* row_theta_col_theta, CUDA_FLOAT_REAL* row_theta_col_f, CUDA_FLOAT_REAL* row_f_col_theta, CUDA_FLOAT_REAL* row_f_col_f){
    int out_index = get_global_index();
    if(out_index < columns*rows*matrices) {

        int current_col = 0, current_row = 0, current_mat = 0;
        get_current_matrix_indices(current_col, current_row, current_mat, out_index, columns, rows, matrices);

        CUDA_FLOAT out_theta;
        out_theta.x = 0.;
        out_theta.y = 0.;
        CUDA_FLOAT out_f;
        out_f.x = 0.;
        out_f.y = 0.;

        for(int m1 = 0; m1 < matrices; ++m1) {
            int in_index = current_col + columns * current_row + (columns*rows)*m1;
            int op_index = current_col + columns * current_row + (columns*rows)*(current_mat*matrices+m1);

            // load
            CUDA_FLOAT in_theta = theta_data[in_index];
            CUDA_FLOAT in_f = f_data[in_index];
            CUDA_FLOAT_REAL op_theta_theta = row_theta_col_theta[op_index];
            CUDA_FLOAT_REAL op_theta_f = row_theta_col_f[op_index];
            CUDA_FLOAT_REAL op_f_theta = row_f_col_theta[op_index];
            CUDA_FLOAT_REAL op_f_f = row_f_col_f[op_index];

            // calc
            out_theta.x += op_theta_theta * in_theta.x + op_theta_f * in_f.x;
            out_theta.y += op_theta_theta * in_theta.y + op_theta_f * in_f.y;
            out_f.x += op_f_theta * in_theta.x + op_f_f * in_f.x;
            out_f.y += op_f_theta * in_theta.y + op_f_f * in_f.y;
        }

        // store
        theta_matrix_out_data[out_index] = out_theta;
        f_matrix_out_data[out_index] = out_f;
    }
}


// Shorter form of calculate_theta_f_implicit. For moz=2, the implicit operator is
// diagonal, so that only diag(op) is stored in the fields (see special code in constructor)
// and the operator can be applied without an extra inner loop.
__global__ void calculate_theta_f_implicit_moz2(CUDA_FLOAT* theta_data, CUDA_FLOAT* f_data, CUDA_FLOAT* theta_matrix_out_data, CUDA_FLOAT* f_matrix_out_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL* row_theta_col_theta, CUDA_FLOAT_REAL* row_theta_col_f, CUDA_FLOAT_REAL* row_f_col_theta, CUDA_FLOAT_REAL* row_f_col_f){
    int total_index = get_global_index();
    if(total_index < columns*rows*matrices) {
        // load
        CUDA_FLOAT in_theta = theta_data[total_index];
        CUDA_FLOAT in_f = f_data[total_index];
        CUDA_FLOAT_REAL op_theta_theta = row_theta_col_theta[total_index];
        CUDA_FLOAT_REAL op_theta_f = row_theta_col_f[total_index];
        CUDA_FLOAT_REAL op_f_theta = row_f_col_theta[total_index];
        CUDA_FLOAT_REAL op_f_f = row_f_col_f[total_index];

        // calc
        CUDA_FLOAT out_theta;
        CUDA_FLOAT out_f;
        out_theta.x = op_theta_theta * in_theta.x + op_theta_f * in_f.x;
        out_theta.y = op_theta_theta * in_theta.y + op_theta_f * in_f.y;
        out_f.x = op_f_theta * in_theta.x + op_f_f * in_f.x;
        out_f.y = op_f_theta * in_theta.y + op_f_f * in_f.y;

        // store
        theta_matrix_out_data[total_index] = out_theta;
        f_matrix_out_data[total_index] = out_f;
    }
}


__global__ void calculate_g_implicit(CUDA_FLOAT* g_data, CUDA_FLOAT* g_matrix_out_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL prandtl_number, CUDA_FLOAT_REAL delta_t) {
    int total_index = get_global_index();
    int current_col = 0, current_row = 0, current_mat = 0;
    get_current_matrix_indices(current_col, current_row, current_mat, total_index, columns, rows, matrices);

	//get x-mode
    CUDA_FLOAT_REAL x_mode = get_wave_number_x(current_col, columns, cube_length_x);
	//get y-mode
    CUDA_FLOAT_REAL y_mode = get_wave_number_y(current_row, rows, cube_length_y);
    // get square of absolute value of mode
    CUDA_FLOAT_REAL mode_sq = x_mode*x_mode + y_mode*y_mode;

	if(total_index < columns*rows*matrices) {

        CUDA_FLOAT out_g;
        if(mode_sq != 0.0){

            // load
            CUDA_FLOAT in_g = g_data[total_index];

            // calc
            // L = 0.5 * mode_abs * (mode_abs + pow2((current_mat+1) * M_PI))
            // inv_B = -2.0 * prandtl_number / mode_abs
            // calc operator (1-dt*inv_B*L)^(-1)
            CUDA_FLOAT_REAL m_pi = (current_mat+1) * M_PI;
            CUDA_FLOAT_REAL factor = 1.0 / (1.0 + delta_t * prandtl_number * (mode_sq + m_pi*m_pi));

            // apply inverse operator
            out_g.x = factor * in_g.x;
            out_g.y = factor * in_g.y;
		} else {
            out_g.x = 0.0;
            out_g.y = 0.0;
		}

        // store
        g_matrix_out_data[total_index] = out_g;
	}

}


__global__ void calculate_F_G_implicit(CUDA_FLOAT* F_data, CUDA_FLOAT* G_data, CUDA_FLOAT* F_matrix_out_data, CUDA_FLOAT* G_matrix_out_data, int columns, int rows, int matrices, CUDA_FLOAT_REAL prandtlNumber, CUDA_FLOAT_REAL delta_t){
    int total_index = get_global_index();
    int col_index = 0, row_index = 0, mat_index = 0;
    get_current_matrix_indices(col_index, row_index, mat_index, total_index, columns, rows, matrices);

	if(total_index < columns*rows*matrices) {

        // load
        CUDA_FLOAT in_F = F_data[total_index];
        CUDA_FLOAT in_G = G_data[total_index];

        // calc
        // L = -0.5*pow2((current_mat+1)*M_PI)
        // B_inv = 2*prandtl_number
        // calc operator (1-dt*B_inv*L)^(-1)
        CUDA_FLOAT_REAL m_pi = (mat_index+1) * M_PI;
        CUDA_FLOAT_REAL factor = 1.0 / (1.0 + delta_t * prandtlNumber * m_pi*m_pi);

        CUDA_FLOAT out_F, out_G;
        out_F.x = factor * in_F.x;
        out_F.y = 0.0;
        out_G.x = factor * in_G.x;
        out_G.y = 0.0;

        // store
        F_matrix_out_data[total_index] = out_F;
        G_matrix_out_data[total_index] = out_G;
	}

}






