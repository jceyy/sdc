#include "calculate_velocity_operator.h"


calculate_velocity_operator* calculate_velocity_operator::init(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length) {
	calculate_velocity_operator* op = new calculate_velocity_operator(dimension, cube_length);
	return op;	
}


/***********************************************************************************/
/** Constructor                                                                   **/
/***********************************************************************************/
calculate_velocity_operator::calculate_velocity_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length){

    //get length of the cube
    cube_length_x = cube_length.at(0);
    cube_length_y = cube_length.at(1);
    cube_length_z = cube_length.at(2);

    //create fft plan
    int num_x = 2*(dimension.at(0)-1);
    int num_y = dimension.at(1);
    cufftPlan2d(&c2r_plan, num_y, num_x, CUFFT_C2R);
}

calculate_velocity_operator::~calculate_velocity_operator(){
	cufftDestroy(c2r_plan);
}


/***********************************************************************************/
/** Calculate velocity operator                                                   **/
/** num_z must be at least 2 for the algorithm to work, and at least 3 for        **/
/** reasonable results                                                            **/
/***********************************************************************************/
matrix_folder_real* calculate_velocity_operator::calculate_operator(matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G, int number_of_z_points) {

    matrix_folder_real* derivs = calculate_derivative_matrix(f, g, F, G);

    //add them up with the right values at vertical points
    vector<int> f_dim = f->get_matrix(0)->get_matrix_dimension();
    int num_x_real = 2*(f_dim.at(0)-1);
    int num_y_real = f_dim.at(1);

    //set return values
    std::vector<int> real_dimensions(3);
    real_dimensions[0] = num_x_real; real_dimensions[1] = num_y_real; real_dimensions[2] = number_of_z_points;
    matrix_device_real* u_1 = new matrix_device_real(real_dimensions);
    matrix_device_real* u_2 = new matrix_device_real(real_dimensions);
    matrix_device_real* u_3 = new matrix_device_real(real_dimensions);

    CUDA_FLOAT_REAL *linspace_z;
    cudaMalloc((void**) &linspace_z, sizeof(CUDA_FLOAT_REAL) * number_of_z_points);
    linspace<<<create_grid_dim(number_of_z_points),create_block_dim(number_of_z_points)>>>
            (number_of_z_points, linspace_z, -0.5, 0.5);

    calculate_operator_at(linspace_z, number_of_z_points, derivs, u_1->get_data(), u_2->get_data(), u_3->get_data());

    delete derivs;// added 09Jul13
    cudaFree(linspace_z);

    matrix_folder_real* u = new matrix_folder_real(3);
    u->add_matrix(0, u_1);
    u->add_matrix(1, u_2);
    u->add_matrix(2, u_3);
    return u;
}


/***********************************************************************************/
/** Calculate velocity operator from derivatives                                  **/
/***********************************************************************************/
matrix_folder_real* calculate_velocity_operator::calculate_derivative_matrix(matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G) {

    vector<int> u_dim(3);
    u_dim[0] = 2*(f->get_matrix(0)->get_matrix_size(0) - 1);
    u_dim[1] = f->get_matrix(0)->get_matrix_size(1);
    u_dim[2] = f->get_matrix(0)->get_matrix_size(2);

    // create complex matrices
    matrix_device_real *u_1_1 = new matrix_device_real(u_dim);
    matrix_device_real *u_2_1 = new matrix_device_real(u_dim);
    matrix_device_real *u_3_1 = new matrix_device_real(u_dim);
    matrix_device_real *u_1_2 = new matrix_device_real(u_dim);
    matrix_device_real *u_2_2 = new matrix_device_real(u_dim);

    // Calculate derivatives (real-valued)
    calculate_derivatives(f, g, F, G, u_1_1->get_data(), u_2_1->get_data(), u_3_1->get_data(), u_1_2->get_data(), u_2_2->get_data());

    matrix_folder_real *u = new matrix_folder_real(5);
    u->add_matrix(0, u_1_1);
    u->add_matrix(1, u_2_1);
    u->add_matrix(2, u_3_1);
    u->add_matrix(3, u_1_2);
    u->add_matrix(4, u_2_2);
    return u;
}


/***********************************************************************************/
/** Calculate velocity operator at a list of z-positions stored in at(0-...) .    **/
/** Needs derivatives, as returned by calculate_derivatives                       **/
/** Order: u_1_1, u_2_1, u_3_1, u_1_2, u_2_2                                      **/
/** u_123 must be of size derivs.dim[0] x derivs.dim[1] x at_size.                **/
/***********************************************************************************/
void calculate_velocity_operator::calculate_operator_at(CUDA_FLOAT_REAL *at_z_device, int size_z, matrix_folder_real *derivs,
                                                        CUDA_FLOAT_REAL *u_1, CUDA_FLOAT_REAL *u_2, CUDA_FLOAT_REAL *u_3) {

    // get derivs data
    vector<int> dim = derivs->get_matrix(0)->get_matrix_dimension();
    CUDA_FLOAT_REAL* u_1_1 = derivs->get_matrix(0)->get_data();
    CUDA_FLOAT_REAL* u_2_1 = derivs->get_matrix(1)->get_data();
    CUDA_FLOAT_REAL* u_3_1 = derivs->get_matrix(2)->get_data();
    CUDA_FLOAT_REAL* u_1_2 = derivs->get_matrix(3)->get_data();
    CUDA_FLOAT_REAL* u_2_2 = derivs->get_matrix(4)->get_data();

    // Create output data fields
    int num_ansatz = dim.at(2);
    int size_xy = dim.at(0) * dim.at(1);
    long size_xyz = long(size_xy) * long(size_z);

    // Calculate all coefficients of vertical ansatz functions
    int num_coeff = num_ansatz * size_z;
    CUDA_FLOAT_REAL* Ci;
    CUDA_FLOAT_REAL* Si;
    CUDA_FLOAT_REAL* dzCi;
    cudaMalloc((void**) &Ci,   sizeof(CUDA_FLOAT_REAL) * num_coeff);
    cudaMalloc((void**) &Si,   sizeof(CUDA_FLOAT_REAL) * num_coeff);
    cudaMalloc((void**) &dzCi, sizeof(CUDA_FLOAT_REAL) * num_coeff);
    dim3 grid_dim_coeff  = create_grid_dim(num_coeff);
    dim3 block_dim_coeff = create_block_dim(num_coeff);
    get_z_coeff_at<<<grid_dim_coeff, block_dim_coeff>>>(at_z_device, size_z, num_ansatz, Ci, Si, dzCi);

    // Compose u from derivative fields and ansatz coefficients
    dim3 grid_dim = create_grid_dim(size_xyz);
    dim3 block_dim = create_block_dim(size_xyz);
    compose_u_from_derivs<<<grid_dim, block_dim>>>(u_1_1, u_2_1, u_3_1, u_1_2, u_2_2,
                                              Ci, Si, dzCi, num_ansatz,
                                              size_xy, size_z, u_1, u_2, u_3);

    // Free data
    cudaFree(Ci);
    cudaFree(Si);
    cudaFree(dzCi);

    DBGSYNC();
}

// Wrapper function to simplify operator calculation at specified z-positions.
// TODO: This operator should be fixed, so that the relevant functions are more simple!
void calculate_velocity_operator::calculate_operator_at(CUDA_FLOAT_REAL *at_z_device, int at_size,
                           matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G,
                           CUDA_FLOAT_REAL *u_1, CUDA_FLOAT_REAL *u_2, CUDA_FLOAT_REAL *u_3) {

    matrix_folder_real* derivs = calculate_derivative_matrix(f, g, F, G);
    calculate_operator_at(at_z_device, at_size, derivs, u_1, u_2, u_3);
    delete derivs;
}

/***********************************************************************************/
/** Calculation of derivatives                                                    **/
/***********************************************************************************/
/*!
* Calculate derivatives:
    u_1_1 = FFT(df/dx)
    u_2_1 = FFT(df/dy)
    u_3_1 = FFT(Laplace(x,y) f)
    u_1_2 = FFT(dg/dy)
    u_2_2 = FFT(dg/dx)
*/
// u_i_j must be of size sizeof(CUDA_FLOAT_REAL) * 2*(f_dim.at(0) -1) * f_dim.at(1) * f_dim.at(2)
void calculate_velocity_operator::calculate_derivatives(matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G,
                    CUDA_FLOAT_REAL* u_1_1, CUDA_FLOAT_REAL* u_2_1, CUDA_FLOAT_REAL* u_3_1, CUDA_FLOAT_REAL* u_1_2, CUDA_FLOAT_REAL* u_2_2){

    matrix_device* f_device = f->get_matrix(0);
    std::vector<int> f_dim = f_device->get_matrix_dimension();
    const int num_x = f_dim.at(0);
    const int num_y = f_dim.at(1);
    const int num_z = f_dim.at(2);
    int num_elements_real = 2*(num_x-1) * num_y;
    dim3 grid_dim;
    dim3 block_dim;

    //part for f
    grid_dim = f_device->create_grid_dimension();
    block_dim = f_device->create_block_dimension();
    int f_dim01 = num_x * num_y;

	//...build \partial_x f(q_1, q_2, n, t)	
    matrix_device* d_f_d_x = new matrix_device(f_device, matrix_device::noInit);
	CUDA_FLOAT* d_f_d_x_data = d_f_d_x->get_data();
    create_derivative_real_matrix_x<<<grid_dim, block_dim>>>(f_device->get_data(), d_f_d_x_data, num_x, num_y, num_z, cube_length_x, 1.0);

	//...build \partial_y f(q_1, q_2, n, t)	
    matrix_device* d_f_d_y = new matrix_device(f_device, matrix_device::noInit);
	CUDA_FLOAT* d_f_d_y_data = d_f_d_y->get_data();
    create_derivative_real_matrix_y<<<grid_dim, block_dim>>>(f_device->get_data(), d_f_d_y_data, num_x, num_y, num_z, cube_length_y, 1.0);

	//...build ( \partial_x^2 + \partial_y^2 ) * f(q_1, q_2, n, t)	
    matrix_device* laplace_f = new matrix_device(f_device, matrix_device::noInit);
	CUDA_FLOAT* laplace_f_data = laplace_f->get_data();
    create_laplace_real_matrix<<<grid_dim, block_dim>>>(f_device->get_data(), laplace_f_data, num_x, num_y, num_z, cube_length_x, cube_length_y, -1.0);

    //...now build u_1^{(1)}(x,y,n,t)
    for(int i = 0; i < num_z; i++) {
        CUFFT_EXEC_C2R(c2r_plan, d_f_d_x_data + i * f_dim01, u_1_1 + i * num_elements_real);
    }
    //...now build u_2^{(1)}(x,y,n,t)
    for(int i = 0; i < num_z; i++) {
        CUFFT_EXEC_C2R(c2r_plan, d_f_d_y_data + i * f_dim01, u_2_1 + i * num_elements_real);
	}
    //...now build u_3^{(1)}(x,y,n,t)
    for(int i = 0; i < num_z; i++) {
        CUFFT_EXEC_C2R(c2r_plan, laplace_f_data + i * f_dim01, u_3_1 + i * num_elements_real);
	}
	
	//part for g
    matrix_device *g_device = g->get_matrix(0);
	//...build \partial_y g(q_1, q_2, n, t)
    matrix_device* d_g_d_y = new matrix_device(g_device, matrix_device::noInit);
	CUDA_FLOAT* d_g_d_y_data = d_g_d_y->get_data();
    create_derivative_real_matrix_y<<<grid_dim, block_dim>>>(g_device->get_data(), d_g_d_y_data, num_x, num_y, num_z, cube_length_y, 1.0);
    add_dc_part<<<grid_dim, block_dim>>>(d_g_d_y_data, F->get_matrix(0)->get_data(), num_x, num_y, num_z);

	//...build (-1) * \partial_x g(q_1, q_2, n, t)
    matrix_device* d_g_d_x = new matrix_device(g->get_matrix(0), matrix_device::noInit);
	CUDA_FLOAT* d_g_d_x_data = d_g_d_x->get_data();
    create_derivative_real_matrix_x<<<grid_dim, block_dim>>>(g_device->get_data(), d_g_d_x_data, num_x, num_y, num_z, cube_length_x, -1.0);
    add_dc_part<<<grid_dim, block_dim>>>(d_g_d_x_data, G->get_matrix(0)->get_data(), num_x, num_y, num_z);
	
    //...now build u_1^{(2)}(x,y,n,t)
    for(int i = 0; i < num_z; i++) {
        CUFFT_EXEC_C2R(c2r_plan, d_g_d_y_data + i * f_dim01, u_1_2 + i * num_elements_real);
	}
    //...now build u_2^{(2)}(x,y,n,t)
    for(int i = 0; i < num_z; i++) {
        CUFFT_EXEC_C2R(c2r_plan, d_g_d_x_data + i * f_dim01, u_2_2 + i * num_elements_real);
	}

    delete d_f_d_x;
    delete d_f_d_y;
    delete laplace_f;
    delete d_g_d_y;
    delete d_g_d_x;
}



/***********************************************************************************/
/** CUDA functions                                                                **/
/***********************************************************************************/
__host__ static dim3 create_block_dim(long number_of_matrix_entries){
    dim3 block;
    block.x = MAX_NUMBER_THREADS_PER_BLOCK;
    return block;
}

__host__ static dim3 create_grid_dim(long number_of_matrix_entries){
    dim3 grid;
    grid.x = (number_of_matrix_entries + MAX_NUMBER_THREADS_PER_BLOCK - 1) / MAX_NUMBER_THREADS_PER_BLOCK;

    if(grid.x > 65535) {
        //EXIT_ERROR("Grid dimension too large.");
        // This would work, but 65k blocks is not recommended:
        while(grid.x > 65535) {
            grid.x = (grid.x + 1) / 2;
            grid.y *= 2;
        }
        while(grid.y > 65535) {
            grid.y = (grid.y + 1) / 2;
            grid.z *= 2;
        }
        if(grid.z > 65535) EXIT_ERROR("Grid dimension too large.");
        //grid.y = 1 + grid.x / 65535;
        //grid.x = 65535;
    }

    return grid;
}


__device__ static int get_global_index() {
    return (threadIdx.x + (threadIdx.y + (threadIdx.z + (blockIdx.x + (blockIdx.y + (blockIdx.z)
            * gridDim.y) * gridDim.x) * blockDim.z) * blockDim.y) * blockDim.x);
}

__device__ static void get_current_matrix_indices(int& col_index, int& row_index, int& mat_index,
                                                  int total_index, int columns, int rows, int matrices) {

    int xysize = rows * columns;

    col_index = (total_index % columns);
    row_index = ((total_index % xysize) / columns);
    mat_index = ((total_index % (xysize * matrices)) / xysize);
}

// function lambda not in coeff.h, but on device for maximum efficiency
// k starts from 1
__device__ static CUDA_FLOAT_REAL lambda(int k)
{
    if((unsigned int) k > 8u) {
        return M_PI * (k + 0.5);
    }

    const CUDA_FLOAT_REAL exact_roots[] =
        {0.0,
         1.505618731142,
         2.499752670074,
         3.500010679436,
         4.499999538484,
         5.500000019944,
         6.499999999138,
         7.500000000037,
         8.499999999998};

    return M_PI * exact_roots[k];
}

// linear spacing of (entries) values, between and including min and max
__global__ static void linspace(int entries, CUDA_FLOAT_REAL* positions,
                                CUDA_FLOAT_REAL min, CUDA_FLOAT_REAL max) {
    int index = get_global_index();

    if(index < entries){
        positions[index] = min + index * (max - min) / (entries - 1);
    }
}

// output(k,l,m) = i*k*factor*input(k,l,m)
__global__ void create_derivative_real_matrix_x(CUDA_FLOAT* input, CUDA_FLOAT* output,
                                                int columns, int rows, int matrices,
                                                CUDA_FLOAT_REAL cube_length_x,
                                                CUDA_FLOAT_REAL factor){
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices){
        // load
        CUDA_FLOAT inval = input[total_index];

        // calc
        int col_index = 0, row_index = 0, mat_index = 0;
        get_current_matrix_indices(col_index, row_index, mat_index, total_index, columns, rows, matrices);
        // derivative is ik
        factor *= col_index * (2*M_PI / cube_length_x);
        CUDA_FLOAT outval;
        outval.x = -inval.y * factor;
        outval.y = inval.x * factor;

        // store
        output[total_index] = outval;
    }
}

// output(k,l,m) = i*l*factor*input(k,l,m)
__global__ void create_derivative_real_matrix_y(CUDA_FLOAT* input, CUDA_FLOAT* output,
                                                int columns, int rows, int matrices,
                                                CUDA_FLOAT_REAL cube_length_y,
                                                CUDA_FLOAT_REAL factor){
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices){
        // load
        CUDA_FLOAT inval = input[total_index];

        // calc
        int col_index = 0, row_index = 0, mat_index = 0;
        get_current_matrix_indices(col_index, row_index, mat_index, total_index, columns, rows, matrices);
        // derivative is ik
        if(row_index < (rows/2)+1)
            factor *= row_index * (2*M_PI / cube_length_y);
        else
            factor *= (row_index - rows) * (2*M_PI / cube_length_y);
        CUDA_FLOAT outval;
        outval.x = -inval.y * factor;
        outval.y = inval.x * factor;

        // store
        output[total_index] = outval;
    }
}


// output(k,l,m) = -(k^2+l^2)*factor*input(k,l,m)
__global__ void create_laplace_real_matrix(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                           CUDA_FLOAT_REAL factor){
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices){
        int col_index = 0, row_index = 0, mat_index = 0;
        get_current_matrix_indices(col_index, row_index, mat_index, total_index, columns, rows, matrices);

        // load
        CUDA_FLOAT val = input[total_index];

        // calc
        CUDA_FLOAT_REAL mode_x = col_index * (2*M_PI / cube_length_x);
        CUDA_FLOAT_REAL mode_y;
        if(row_index < (rows/2)+1)
            mode_y = row_index * (2*M_PI / cube_length_y);
        else
            mode_y = (row_index - rows) * (2*M_PI / cube_length_y);

        factor *= (mode_x*mode_x + mode_y*mode_y);
        val.x *= -factor;
        val.y *= -factor;

        // store
        output[total_index] = val;
    }

}

/***********************************************************************************/
/** Compose velocity field u from derivatives u_i_j of size Nx x Ny x Npart.      **/
/***********************************************************************************/
/*!
    u_1_1 = FFT(df/dx)
    u_2_1 = FFT(df/dy)
    u_3_1 = FFT(Laplace(x,y) f)
    u_1_2 = FFT(dg/dy)
    u_2_2 = FFT(dg/dx)
*/
__global__ void compose_u_from_derivs(CUDA_FLOAT_REAL* u_1_1, CUDA_FLOAT_REAL* u_2_1, CUDA_FLOAT_REAL* u_3_1,
                                      CUDA_FLOAT_REAL* u_1_2, CUDA_FLOAT_REAL* u_2_2,
                                      CUDA_FLOAT_REAL* Ci, CUDA_FLOAT_REAL* Si, CUDA_FLOAT_REAL* dzCi, int num_ansatz,
                                      int xy_size, int z_size, CUDA_FLOAT_REAL* u_1, CUDA_FLOAT_REAL* u_2, CUDA_FLOAT_REAL* u_3) {
    int index = get_global_index();
    int total_size = xy_size * z_size;

    if(index < total_size) {
        int xy_index = index % xy_size;
        int z_index = index / xy_size;

        // sum up ansatz coefficients * functions
        CUDA_FLOAT_REAL u1 = 0.0;
        CUDA_FLOAT_REAL u2 = 0.0;
        CUDA_FLOAT_REAL u3 = 0.0;

        for(int i = 0; i < num_ansatz; i++) {
            int u_index = i*xy_size+xy_index;
            int coeff_index = z_index*num_ansatz + i;

            u1 += dzCi[coeff_index] * u_1_1[u_index] + Si[coeff_index] * u_1_2[u_index];
            u2 += dzCi[coeff_index] * u_2_1[u_index] + Si[coeff_index] * u_2_2[u_index];
            u3 +=   Ci[coeff_index] * u_3_1[u_index];
        }

        u_1[index] = u1;
        u_2[index] = u2;
        u_3[index] = u3;
    }
}

// Ci is [ C0(z0) C1(z0) .. C0(z1) C1(z1) ... ] ; C = Ci[num_ansatz*z+i_ansatz]
__global__ void get_z_coeff_at(CUDA_FLOAT_REAL* at_z, int z_size, int num_ansatz, CUDA_FLOAT_REAL* Ci, CUDA_FLOAT_REAL* Si, CUDA_FLOAT_REAL* dzCi) {
    int index = get_global_index();

    if(index < num_ansatz * z_size) {
        int m = (index % num_ansatz) + 1;
        int i_at_z = index / num_ansatz;

        CUDA_FLOAT_REAL z = at_z[i_at_z];
        if(z < -0.5) z = -0.5;
        else if(z > 0.5) z = 0.5;

        Si[index] = sin(M_PI*m*(z+0.5));
        CUDA_FLOAT_REAL lambda_m = lambda(m);
        CUDA_FLOAT_REAL lambda_z = lambda_m * z;
        if(m%2==1) {
            // Chandrasekhar function (and derivative)
            // for odd m
            CUDA_FLOAT_REAL inv_cosh = 1.0 / cosh(lambda_m*0.5);
            CUDA_FLOAT_REAL inv_cos = 1.0 / cos(lambda_m*0.5);
            Ci[index]   = cosh(lambda_z)*inv_cosh - cos(lambda_z)*inv_cos;
            dzCi[index] = lambda_m*(sinh(lambda_z)*inv_cosh + sin(lambda_z)*inv_cos);
        } else {
            // Chandrasekhar function (and derivative)
            // for even m
            CUDA_FLOAT_REAL inv_sinh = 1.0 / sinh(lambda_m*0.5);
            CUDA_FLOAT_REAL inv_sin = 1.0 / sin(lambda_m*0.5);
            Ci[index]   = sinh(lambda_z)*inv_sinh - sin(lambda_z)*inv_sin;
            dzCi[index] = lambda_m*(cosh(lambda_z)*inv_sinh - cos(lambda_z)*inv_sin);
        }
    }
}


__global__ void add_dc_part(CUDA_FLOAT* g, CUDA_FLOAT* F, int columns, int rows, int matrices) {
    int total_index = get_global_index();
    int col_index = 0, row_index = 0, m_index = 0;
    get_current_matrix_indices(col_index, row_index, m_index, total_index, columns, rows, matrices);

    // check if thread is valid and at mode 0
    // this could be simplified by only calling moz threads, not mox*moy*moz with if!
    if(total_index < columns*rows*matrices && col_index == 0 && row_index == 0){
        CUDA_FLOAT g_val = g[total_index];
        CUDA_FLOAT F_val = F[m_index];
        g_val.x += F_val.x;
        g_val.y += F_val.y;
        g[total_index] = g_val;
    }
}





