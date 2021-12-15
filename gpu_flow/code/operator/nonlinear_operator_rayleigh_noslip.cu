#include "nonlinear_operator_rayleigh_noslip.h"

// BUGS:
// found 12.03.2013: matrix d_C_iC_j_C_1_host is not -raab
// found 14.03.2013: addup_only_dc_part on F is called for dim_xy threads, while it takes dim_xyz threads to complete the operation. However, with dim_xyz threads, F gets complex
// found 15.03.2013: in building laplace_f, only the first moy*moy*moz elements were *(-1)'d
// found 07.06.2013: d_C_iC_j_C_1_host is -rbbb, changed back
// found 10.06.2013: in summand  8: coeff.Iabc(i,j,k) is coeff.Iabc(j,i,k)
// found 19.08.2013: in summand 17: coeff.Iabc(i,j,k) is coeff.Iabc(j,i,k)

nonlinear_operator_rayleigh_noslip::nonlinear_operator_rayleigh_noslip(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL prandtl_Number, Coeff<double>& p_coeff, CUDA_FLOAT_REAL penalization_eta) : coeff(p_coeff){

    //get length of the cube
    cube_length_x = cube_length.at(0);
    cube_length_y = cube_length.at(1);
    cube_length_z = cube_length.at(2);

    //set prandtl number
    prandtl_number = prandtl_Number;
#ifdef PENALIZED
    if(penalization_eta <= 0.0) EXIT_ERROR("penalization eta must be positive.");
    eta = penalization_eta;
#endif /* PENALIZED */

    // Number of elements in real-space (after FFT)
    const int num_x = 2*(dimension.at(0)-1);
    const int num_y = dimension.at(1);

#ifdef PENALIZED
    int num_elements_real = num_x * num_y;
    cudaMalloc((void**) &mask, sizeof(CUDA_FLOAT_REAL) * num_elements_real );
    dim3 gridDim_mask = create_grid_dim(num_elements_real);
    dim3 blockDim_mask = create_block_dim(num_elements_real);

    //use other init mask
#ifdef CIRCLE
    init_mask_physical_space_circle<<<gridDim_mask,blockDim_mask>>>(mask, num_x, num_y);
#else
#ifdef DISKARRAY
    init_mask_physical_space_disk_array<<<gridDim_mask,blockDim_mask>>>(mask, num_x, num_y, 3, 2);
#else
    init_mask_physical_space_rectangle<<<gridDim_mask,blockDim_mask>>>(mask, num_x, num_y);
    //DBGOUT("Mask has been changed.");
    //init_mask_physical_space_everywhere<<<gridDim_mask,blockDim_mask>>>(mask, num_x, num_y);
#endif
#endif

#ifdef DEBUG
    CUDA_FLOAT_REAL* tmp_host = new CUDA_FLOAT_REAL[num_elements_real];
    cudaMemcpy(tmp_host, mask , sizeof(CUDA_FLOAT_REAL) * num_elements_real, cudaMemcpyDeviceToHost);
    std::cout << "number of real elements: " << num_elements_real << std::endl;
    std::cout << "in x direction " << num_elements_real_x << std::endl;
    std::cout << "in y direction " << num_elements_real_y << std::endl;
    for(int k = 0; k < num_elements_real_y; k++){
        for(int l = 0; l < num_elements_real_x; l++)
            std::cout << tmp_host[k*num_elements_real_x + l] << " ";
        std::cout << std::endl;
    }
    delete tmp_host;
#endif

#endif /* PENALIZED */

    //create fft plan
    cufftPlan2d(&c2r_plan, num_y, num_x, CUFFT_TYPE_C2R);
    //create inverse fft plan
    cufftPlan2d(&r2c_plan, num_y, num_x, CUFFT_TYPE_R2C);
}


nonlinear_operator_rayleigh_noslip::~nonlinear_operator_rayleigh_noslip(){
    //free cuda fft plans
    cufftDestroy(c2r_plan);
    cufftDestroy(r2c_plan);
}


nonlinear_operator_rayleigh_noslip* nonlinear_operator_rayleigh_noslip::init_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length, CUDA_FLOAT_REAL prandtl_Number, Coeff<double> &p_coeff, cufftReal penalization_eta){
    nonlinear_operator_rayleigh_noslip* op = new nonlinear_operator_rayleigh_noslip(dimension, cube_length, prandtl_Number, p_coeff, penalization_eta);
    return op;
}

matrix_folder** nonlinear_operator_rayleigh_noslip::calculate_operator(matrix_folder* theta, matrix_folder* f, matrix_folder* g, matrix_folder* F, matrix_folder* G){

    #ifdef DEBUG
        DBGOUT("calculate nonlinear operator");
    #endif

    matrix_folder** return_array = new matrix_folder*[5];

    #ifdef DEBUG
        DBGOUT("transform to physical space");
    #endif

    // build u_i^{(1)}(x,y,n,t) for i = 1,2,3
    std::vector<int> f_dim = f->get_matrix(0)->get_matrix_dimension();
    const int mox = f_dim.at(0),    // modes in x direction
              moy = f_dim.at(1),    // modes in y direction
              moz = f_dim.at(2);    // modes in z direction
    const int moxy = mox * moy;     // modes in horizontal plane
    const int num_elements_real = 2 * (mox-1) * moy;    // number of elements in real space

    const dim3 grid_moxy = create_grid_dim(moxy);
    const dim3 block_moxy = create_block_dim(moxy);
    const dim3 grid_moz = create_grid_dim(moz);
    const dim3 block_moz = create_block_dim(moz);
    const dim3 grid_f = create_grid_dim(moxy*moz);
    const dim3 block_f = create_block_dim(moxy*moz);
    const dim3 grid_f_real = create_grid_dim(num_elements_real);
    const dim3 block_f_real = create_block_dim(num_elements_real);

    //...build \partial_x f(q_1, q_2, n, t)
    matrix_device* d_f_d_x = new matrix_device(f->get_matrix(0), matrix_device::noInit);
    CUDA_FLOAT* d_f_d_x_data = d_f_d_x->get_data();
    create_derivative_x<<<grid_f,block_f>>>(f->get_matrix(0)->get_data(), d_f_d_x_data, mox, moy, moz, cube_length_x);
    //...build \partial_y f(q_1, q_2, n, t)
    matrix_device* d_f_d_y = new matrix_device(f->get_matrix(0), matrix_device::noInit);
    CUDA_FLOAT* d_f_d_y_data = d_f_d_y->get_data();
    create_derivative_y<<<grid_f,block_f>>>(f->get_matrix(0)->get_data(), d_f_d_y_data, mox, moy, moz, cube_length_y);
    //...build ( \partial_x^2 + \partial_y^2 ) * f(q_1, q_2, n, t)
    matrix_device* laplace_f = new matrix_device(f->get_matrix(0), matrix_device::noInit);
    CUDA_FLOAT* laplace_f_data = laplace_f->get_data();
    create_laplace_xy<<<grid_f,block_f>>>(f->get_matrix(0)->get_data(), laplace_f_data, mox, moy, moz, cube_length_x, cube_length_y);
    scale_on_device<<<grid_f,block_f>>>(laplace_f_data, -1.0, laplace_f_data, moxy*moz);
    //...now build u_1^{(1)}(x,y,n,t)
    CUDA_FLOAT_REAL* u_1_1;

    if(cudaSuccess != cudaMalloc((void**) &u_1_1, sizeof(CUDA_FLOAT_REAL) * moz * num_elements_real) ) {
        EXIT_ERROR("not able to allocate transformation memory");
    }
    DBGSYNC();
    for(int i = 0; i < moz; i++) {
        if(CUFFT_SUCCESS != CUFFT_EXEC_C2R(c2r_plan, d_f_d_x_data+i*(moxy), u_1_1 + i*num_elements_real) ) {
            DBGSYNC();
            EXIT_ERROR2("c2r-fft failed", ::cudaGetErrorString(cudaGetLastError()));
        }
    }
    delete d_f_d_x;
    //...now build u_2^{(1)}(x,y,n,t)
    CUDA_FLOAT_REAL* u_2_1;
    cudaMalloc((void**) &u_2_1, sizeof(CUDA_FLOAT_REAL) * moz * num_elements_real);
    for(int i = 0; i < moz; i++) {
        if(CUFFT_SUCCESS != CUFFT_EXEC_C2R(c2r_plan, d_f_d_y_data + i*moxy, u_2_1 + i*num_elements_real)) {
            EXIT_ERROR2("c2r-fft failed", ::cudaGetErrorString(cudaGetLastError()));
        }
    }
    delete d_f_d_y;
    //...now build u_3^{(1)}(x,y,n,t)
    CUDA_FLOAT_REAL* u_3_1;
    cudaMalloc((void**) &u_3_1, sizeof(CUDA_FLOAT_REAL) * moz * num_elements_real);
    for(int i = 0; i < moz; i++) {
        if(CUFFT_SUCCESS != CUFFT_EXEC_C2R(c2r_plan, laplace_f_data + i*moxy, u_3_1 + i*num_elements_real)) {
            EXIT_ERROR2("c2r-fft failed", ::cudaGetErrorString(cudaGetLastError()));
        }
    }
    delete laplace_f;

    //build u_i^{(2)}(x,y,n,t) for i = 1,2

    //...build \partial_y g(q_1, q_2, n, t)
    matrix_device* d_g_d_y = new matrix_device(g->get_matrix(0), matrix_device::noInit);
    CUDA_FLOAT* d_g_d_y_data = d_g_d_y->get_data();
    create_derivative_y<<<grid_f,block_f>>>(g->get_matrix(0)->get_data(), d_g_d_y_data, mox, moy, moz, cube_length_y);
    add_dc_part<<<grid_moz,block_moz>>>(d_g_d_y_data, F->get_matrix(0)->get_data(), mox, moy, moz);
    //...build (-1) * \partial_x g(q_1, q_2, n, t)
    matrix_device* d_g_d_x = new matrix_device(g->get_matrix(0), matrix_device::noInit);
    CUDA_FLOAT* d_g_d_x_data = d_g_d_x->get_data();
    create_derivative_x<<<grid_f,block_f>>>(g->get_matrix(0)->get_data(), d_g_d_x_data, mox, moy, moz, cube_length_x);
    scale_on_device<<<grid_f,block_f>>>(d_g_d_x_data, -1.0, d_g_d_x_data, moxy*moz);
    add_dc_part<<<grid_moz,block_moz>>>(d_g_d_x_data, G->get_matrix(0)->get_data(), mox, moy, moz);

    //...now build u_1^{(2)}(x,y,n,t)
    CUDA_FLOAT_REAL* u_1_2;
    cudaMalloc((void**) &u_1_2, sizeof(CUDA_FLOAT_REAL) * moz * num_elements_real);
    for(int i = 0; i < moz; i++) {
        if(CUFFT_SUCCESS != CUFFT_EXEC_C2R(c2r_plan, d_g_d_y_data+i*moxy, u_1_2 + i*num_elements_real)) {
            EXIT_ERROR2("c2r-fft failed", ::cudaGetErrorString(cudaGetLastError()));
        }
    }
    delete d_g_d_y;
    //...now build u_2^{(2)}(x,y,n,t)
    CUDA_FLOAT_REAL* u_2_2;
    cudaMalloc((void**) &u_2_2, sizeof(CUDA_FLOAT_REAL) * moz * num_elements_real);
    for(int i = 0; i < moz; i++) {
        if(CUFFT_SUCCESS != CUFFT_EXEC_C2R(c2r_plan, d_g_d_x_data+i*moxy, u_2_2 + i*num_elements_real)) {
            EXIT_ERROR2("c2r-fft failed", ::cudaGetErrorString(cudaGetLastError()));
        }
    }
    delete d_g_d_x;

    //build \theta(x,y,i) , \partial_x(\theta(x,y,i)) for i=1,2
    //... copy of theta to perform fourier transformation
    matrix_device* theta_copy = new matrix_device(theta->get_matrix(0), matrix_device::Copy);
    CUDA_FLOAT* theta_data = theta_copy->get_data();
    //.... \partial_x theta in fourier space
    matrix_device* d_x_theta = new matrix_device(theta->get_matrix(0), matrix_device::noInit);
    CUDA_FLOAT* d_x_theta_data = d_x_theta->get_data();
    create_derivative_x<<<grid_f,block_f>>>(theta->get_matrix(0)->get_data(), d_x_theta_data, mox, moy, moz, cube_length_x);
    //.... \partial_y theta in fourier space
    matrix_device* d_y_theta = new matrix_device(theta->get_matrix(0), matrix_device::noInit);
    CUDA_FLOAT* d_y_theta_data = d_y_theta->get_data();
    create_derivative_y<<<grid_f,block_f>>>(theta->get_matrix(0)->get_data(), d_y_theta_data, mox, moy, moz, cube_length_y);

    //...now build \partial_x theta(x,y,i)
    CUDA_FLOAT_REAL* d_x_theta_physical;
    cudaMalloc((void**) &d_x_theta_physical, sizeof(CUDA_FLOAT_REAL) * moz * num_elements_real);
    for(int i = 0; i < moz; i++) {
        #ifdef DEBUG
            DBGOUT("1. execute theta c2r-fft");
        #endif
        CUFFT_EXEC_C2R(c2r_plan, d_x_theta_data+i*(mox*moy) , d_x_theta_physical + i*num_elements_real);
    }
    delete d_x_theta;
    //...now build theta(x,y,i)
    CUDA_FLOAT_REAL* theta_physical;
    cudaMalloc((void**) &theta_physical, sizeof(CUDA_FLOAT_REAL) * moz * num_elements_real);
    for(int i = 0; i < moz; i++) {
        #ifdef DEBUG
            DBGOUT("2. execute theta c2r-fft");
        #endif
        CUFFT_EXEC_C2R(c2r_plan, theta_data +i*(mox*moy) , theta_physical+(i*num_elements_real));
    }
    delete theta_copy;
    //...now build \partial_y theta(x,y,i)
    CUDA_FLOAT_REAL* d_y_theta_physical;
    cudaMalloc((void**) &d_y_theta_physical, sizeof(CUDA_FLOAT_REAL) * moz * num_elements_real);
    for(int i = 0; i < moz; i++) {
        #ifdef DEBUG
            DBGOUT("3. execute theta c2r-fft");
        #endif
        CUFFT_EXEC_C2R(c2r_plan, d_y_theta_data+i*(mox*moy) , d_y_theta_physical + i*num_elements_real);
    }
    delete d_y_theta;

    //now calculate the nonlinear parts of theta f, g, F and G
    //....return data for f,g, theta, F, G and calculation space
    matrix_device* theta_return_data = new matrix_device(f_dim);
    theta_return_data->set_zero();
    matrix_device* f_return_data = new matrix_device(f_dim);
    f_return_data->set_zero();
    matrix_device* g_return_data = new matrix_device(f_dim);
    g_return_data->set_zero();

    std::vector<int> F_dim = F->get_matrix(0)->get_matrix_dimension();
    std::vector<int> G_dim = G->get_matrix(0)->get_matrix_dimension();
    matrix_device* F_return_data = new matrix_device(F_dim);
    F_return_data->set_zero();
    matrix_device* G_return_data = new matrix_device(G_dim);
    G_return_data->set_zero();

    //the number of real samples per vertical ansatz function
    CUDA_FLOAT_REAL fourier_r2c_scaling = 1.0/num_elements_real;

    // Create storage buffer
    CUDA_FLOAT* moxy_complex_buffer;
    cudaMalloc((void**) &moxy_complex_buffer, sizeof(CUDA_FLOAT) * moxy);

    // Create storage buffer
    CUDA_FLOAT_REAL* num_elements_real_buffer;
    cudaMalloc((void**) &num_elements_real_buffer, sizeof(CUDA_FLOAT_REAL) * num_elements_real);

	//...iterate over vertical ansatz functions in a double loop
    for(int j = 0; j < moz; j++){

        //some initial definitions
		//....j
		CUDA_FLOAT_REAL* u_1_1_j = u_1_1+j*num_elements_real;
		CUDA_FLOAT_REAL* u_2_1_j = u_2_1+j*num_elements_real;
		CUDA_FLOAT_REAL* u_3_1_j = u_3_1+j*num_elements_real;
		CUDA_FLOAT_REAL* u_1_2_j = u_1_2+j*num_elements_real;
		CUDA_FLOAT_REAL* u_2_2_j = u_2_2+j*num_elements_real;
		CUDA_FLOAT_REAL* theta_j = theta_physical+j*num_elements_real;
		CUDA_FLOAT_REAL* d_x_theta_j = d_x_theta_physical+j*num_elements_real;
        CUDA_FLOAT_REAL* d_y_theta_j = d_y_theta_physical+j*num_elements_real;

        for(int i = 0; i < moz; i++) {


			//some initial definitions
			//....i
			CUDA_FLOAT_REAL* u_1_1_i = u_1_1+i*num_elements_real;
			CUDA_FLOAT_REAL* u_2_1_i = u_2_1+i*num_elements_real;
			CUDA_FLOAT_REAL* u_3_1_i = u_3_1+i*num_elements_real;
			CUDA_FLOAT_REAL* u_1_2_i = u_1_2+i*num_elements_real;
            CUDA_FLOAT_REAL* u_2_2_i = u_2_2+i*num_elements_real;

			#ifdef DEBUG
                DBGOUT("transform to back to fourier space");
			#endif

			

            //1: calculate \hat{u_1_1_i \cdot u_1_1_j} (q, t)
            CUDA_FLOAT* u_1_1_i_mul_u_1_1_j_fourier;
            cudaMalloc((void**) &u_1_1_i_mul_u_1_1_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_1_i, u_1_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_1_i_mul_u_1_1_j_fourier);

            //calculate \hat{u_1_1_i \cdot u_2_1_j} (q, t)
            CUDA_FLOAT* u_1_1_i_mul_u_2_1_j_fourier;
            cudaMalloc((void**) &u_1_1_i_mul_u_2_1_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_1_i, u_2_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_1_i_mul_u_2_1_j_fourier);

            //calculate \hat{u_1_1_i \cdot u3_1_j} (q, t)
			CUDA_FLOAT* u_1_1_i_mul_u_3_1_j_fourier;
            cudaMalloc((void**) &u_1_1_i_mul_u_3_1_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_1_i, u_3_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_1_i_mul_u_3_1_j_fourier);


            //calculate \hat{u_1_1_i u_1_2_j} (q, t)
            CUDA_FLOAT* u_1_1_i_mul_u_1_2_j_fourier;
            cudaMalloc((void**) &u_1_1_i_mul_u_1_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_1_i, u_1_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_1_i_mul_u_1_2_j_fourier);


            //calculate \hat{u_1_1_i \cdot unum_elements_real_j} (q, t)
            CUDA_FLOAT* u_1_1_i_mul_u_2_2_j_fourier;
            cudaMalloc((void**) &u_1_1_i_mul_u_2_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_1_i, u_2_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_1_i_mul_u_2_2_j_fourier);


            //calculate \hat{u_2_1_i \cdot u_2_1_j} (q, t)
            CUDA_FLOAT* u_2_1_i_mul_u_2_1_j_fourier;
            cudaMalloc((void**) &u_2_1_i_mul_u_2_1_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_2_1_i, u_2_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_2_1_i_mul_u_2_1_j_fourier);


            //calculate \hat{u_2_1_i \cdot u_3_1_j} (q, t)
            CUDA_FLOAT* u_2_1_i_mul_u_3_1_j_fourier;
            cudaMalloc((void**) &u_2_1_i_mul_u_3_1_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_2_1_i, u_3_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_2_1_i_mul_u_3_1_j_fourier);


            //calculate \hat{u_2_1_i \cdot u_1_2_j} (q, t)
            CUDA_FLOAT* u_2_1_i_mul_u_1_2_j_fourier;
            cudaMalloc((void**) &u_2_1_i_mul_u_1_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_2_1_i, u_1_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_2_1_i_mul_u_1_2_j_fourier);


            //calculate \hat{u_2_1_i \cdot u_2_2_j} (q, t)
            CUDA_FLOAT* u_2_1_i_mul_u_2_2_j_fourier;
            cudaMalloc((void**) &u_2_1_i_mul_u_2_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_2_1_i, u_2_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_2_1_i_mul_u_2_2_j_fourier);


            //calculate \hat{u_3_1_i \cdot u_3_1_j} (q, t)
            CUDA_FLOAT* u_3_1_i_mul_u_3_1_j_fourier;
            cudaMalloc((void**) &u_3_1_i_mul_u_3_1_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_3_1_i, u_3_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_3_1_i_mul_u_3_1_j_fourier);

            //calculate \hat{u_3_1_i \cdot u_1_2_j} (q, t)
            CUDA_FLOAT* u_3_1_i_mul_u_1_2_j_fourier;
            cudaMalloc((void**) &u_3_1_i_mul_u_1_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_3_1_i, u_1_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_3_1_i_mul_u_1_2_j_fourier);


            //calculate \hat{u_3_1_i \cdot u_2_2_j} (q, t)
            CUDA_FLOAT* u_3_1_i_mul_u_2_2_j_fourier;
            cudaMalloc((void**) &u_3_1_i_mul_u_2_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_3_1_i, u_2_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_3_1_i_mul_u_2_2_j_fourier);


            //calculate \hat{u_1_2_i \cdot u_1_2_j} (q, t)
            CUDA_FLOAT* u_1_2_i_mul_u_1_2_j_fourier;
            cudaMalloc((void**) &u_1_2_i_mul_u_1_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_2_i, u_1_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_2_i_mul_u_1_2_j_fourier);

            //calculate \hat{u_1_2_i \cdot u_2_2_j} (q, t)
            CUDA_FLOAT* u_1_2_i_mul_u_2_2_j_fourier;
            cudaMalloc((void**) &u_1_2_i_mul_u_2_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_2_i, u_2_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_2_i_mul_u_2_2_j_fourier);


            //calculate \hat{u_2_2_i \cdot u_2_2_j} (q, t)
            CUDA_FLOAT* u_2_2_i_mul_u_2_2_j_fourier;
            cudaMalloc((void**) &u_2_2_i_mul_u_2_2_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_2_2_i, u_2_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_2_2_i_mul_u_2_2_j_fourier);


            //calculate \hat{u_1_1_i \cdot \partial_x(theta)_j} (q, t)
            CUDA_FLOAT* u_1_1_i_mul_d_x_theta_j_fourier;
            cudaMalloc((void**) &u_1_1_i_mul_d_x_theta_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_1_i, d_x_theta_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_1_i_mul_d_x_theta_j_fourier);


            //calculate \hat{u_1_2_i \cdot \partial_x(theta)_j} (q, t)
            CUDA_FLOAT* u_1_2_i_mul_d_x_theta_j_fourier;
            cudaMalloc((void**) &u_1_2_i_mul_d_x_theta_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_1_2_i, d_x_theta_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_1_2_i_mul_d_x_theta_j_fourier);


            //calculate \hat{u_2_1_i \cdot \partial_y(theta)_j} (q, t)
            CUDA_FLOAT* u_2_1_i_mul_d_y_theta_j_fourier;
            cudaMalloc((void**) &u_2_1_i_mul_d_y_theta_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_2_1_i, d_y_theta_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_2_1_i_mul_d_y_theta_j_fourier);


            //calculate \hat{u_2_2_i \cdot \partial_y(theta)_j} (q, t)
            CUDA_FLOAT* u_2_2_i_mul_d_y_theta_j_fourier;
            cudaMalloc((void**) &u_2_2_i_mul_d_y_theta_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_2_2_i, d_y_theta_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_2_2_i_mul_d_y_theta_j_fourier);


            //calculate \hat{u_3_1_i \cdot theta_j} (q, t)
            CUDA_FLOAT* u_3_1_i_mul_theta_j_fourier;
            cudaMalloc((void**) &u_3_1_i_mul_theta_j_fourier, sizeof(CUDA_FLOAT) * moxy );
			//...mult pointwise
            mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(u_3_1_i, theta_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
			//...transform back to fourier space			
            CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, u_3_1_i_mul_theta_j_fourier);

			

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for f//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			#ifdef DEBUG
                DBGOUT("calculate f");
            #endif

            //...summand 1
            create_second_derivative_xx<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_1_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ibbb(i,j,k);// changed from Iaab 07Jun13
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_1 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 2
            create_second_derivative_xx<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ibb1(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, 2*C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_2 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

            //...summand 3
            create_second_derivative_xx<<<grid_moxy,block_moxy>>>(u_1_2_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ib11(k,i,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_3 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

            //...summand 4
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_2_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ibbb(j,i,k);// changed from Iaab 07Jun13
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, 2*C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_4 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 5
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ibb1(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, 2*C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_5 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 6
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ibb1(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, 2*C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_6 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 7
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_1_2_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ib11(k,i,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, 2*C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_7 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

            //...summand 8
            create_derivative_x<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_3_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Iabc(j,i,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_8 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 9
            create_derivative_x<<<grid_moxy,block_moxy>>>(u_3_1_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Iac1(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_9 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

            //...summand 10-13 identical to 4-7
 
			//...summand 14 
            create_second_derivative_yy<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_2_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ibbb(i,j,k);// changed from Iaab 07Jun13
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_14 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 15 
            create_second_derivative_yy<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ibb1(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, 2*C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_15 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 16 
            create_second_derivative_yy<<<grid_moxy,block_moxy>>>(u_2_2_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ib11(k,i,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_16 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 17 
            create_derivative_y<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_3_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Iabc(j,i,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_17 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 18 
            create_derivative_y<<<grid_moxy,block_moxy>>>(u_3_1_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Iac1(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_18 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 19
            create_derivative_x<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_3_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            create_laplace_xy<<<grid_moxy,block_moxy>>>(moxy_complex_buffer, moxy_complex_buffer , mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Iaab(j,k,i);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, -C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_19 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 21
            create_derivative_x<<<grid_moxy,block_moxy>>>(u_3_1_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            create_laplace_xy<<<grid_moxy,block_moxy>>>(moxy_complex_buffer, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Iaa1(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, -C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_21 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 22
            create_derivative_y<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_3_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            create_laplace_xy<<<grid_moxy,block_moxy>>>(moxy_complex_buffer, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Iaab(j,k,i);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, -C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_22 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 23
            create_derivative_y<<<grid_moxy,block_moxy>>>(u_3_1_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            create_laplace_xy<<<grid_moxy,block_moxy>>>(moxy_complex_buffer, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Iaa1(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, -C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_23 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 24
            create_laplace_xy<<<grid_moxy,block_moxy>>>(u_3_1_i_mul_u_3_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Iaab(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, -C, f_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "f_24 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for g//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			#ifdef DEBUG
                DBGOUT("calculate g");
			#endif


			//...summand 1 of g
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_1_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ibb1(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_1 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 2 of g
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ib11(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, 2.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_2 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 3 of g
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_1_2_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.I111(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_3 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 4 of g
            create_second_derivative_yy<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_2_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ibb1(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_4 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 5 of g
            create_second_derivative_yy<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ib11(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_5 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 6 of g
            create_second_derivative_yy<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ib11(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_6 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 7 of g
            create_second_derivative_yy<<<grid_moxy,block_moxy>>>(u_1_2_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.I111(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_7 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 8 of g
            create_derivative_y<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_3_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Iab2(j,i,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_8 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 9 of g
            create_derivative_y<<<grid_moxy,block_moxy>>>(u_3_1_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ia12(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_9 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 10 of g
            create_second_derivative_xx<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_2_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ibb1(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_10 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 11 of g
            create_second_derivative_xx<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ib11(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_11 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 12 of g
            create_second_derivative_xx<<<grid_moxy,block_moxy>>>(u_1_1_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ib11(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_12 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 13 of g
            create_second_derivative_xx<<<grid_moxy,block_moxy>>>(u_1_2_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.I111(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_13 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 14 of g
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_2_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ibb1(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_14 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 15 of g
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ib11(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -2.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_15 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 16 of g
            create_second_derivative_xy<<<grid_moxy,block_moxy>>>(u_2_2_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
            for(int k = 0; k < moz; ++k) {
                double C = coeff.I111(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_16 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 17 of g
            create_derivative_x<<<grid_moxy,block_moxy>>>(u_2_1_i_mul_u_3_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Iab2(j,i,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_17 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 18 of g
            create_derivative_x<<<grid_moxy,block_moxy>>>(u_3_1_i_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ia12(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(g_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0 * C, g_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "g_18 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for F//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			
			#ifdef DEBUG
                DBGOUT("calculate F");
            #endif

            //...summand 1 of F
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Iab2(j,i,k);
                if(C != 0.0) {
                    addup_only_dc_part<<<grid_moz,block_moz>>>(F_return_data->get_data()+k, u_1_1_i_mul_u_3_1_j_fourier, mox, moy, 1, C);
                }
                #ifdef DEBUG
                    cout << "F_1 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

            //...summand 2 of F
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ia12(i,j,k);
                if(C != 0.0) {
                    addup_only_dc_part<<<grid_moz,block_moz>>>(F_return_data->get_data()+k, u_3_1_i_mul_u_1_2_j_fourier, mox, moy, 1, C);
                }
                #ifdef DEBUG
                    cout << "F_2 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for G//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			#ifdef DEBUG
                DBGOUT("calculate G");
			#endif

            //...summand 1 of G
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Iab2(j,i,k);
                if(C != 0.0) {
                    addup_only_dc_part<<<grid_moz,block_moz>>>(G_return_data->get_data()+k, u_2_1_i_mul_u_3_1_j_fourier, mox, moy, 1, C);
                }
                #ifdef DEBUG
                    cout << "G_1 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }


            //...summand 2 of G
            for(int k = 0; k < moz; ++k) {
                double C = -coeff.Ia12(i,j,k);
                if(C != 0.0) {
                    addup_only_dc_part<<<grid_moz,block_moz>>>(G_return_data->get_data()+k, u_3_1_i_mul_u_2_2_j_fourier, mox, moy, 1, C);
                }
                #ifdef DEBUG
                    cout << "G_2 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for theta//////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			#ifdef DEBUG
                DBGOUT("calculate theta");
            #endif

            //...summand 1 of theta
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ib11(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(theta_return_data->get_data() + k*moxy, u_1_1_i_mul_d_x_theta_j_fourier, C, theta_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "theta_1 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 2 of theta
            for(int k = 0; k < moz; ++k) {
                double C = coeff.I111(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(theta_return_data->get_data() + k*moxy, u_1_2_i_mul_d_x_theta_j_fourier, C, theta_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "theta_2 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 3 of theta
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ib11(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(theta_return_data->get_data() + k*moxy, u_2_1_i_mul_d_y_theta_j_fourier, C, theta_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "theta_3 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 4 of theta
            for(int k = 0; k < moz; ++k) {
                double C = coeff.I111(i,j,k);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(theta_return_data->get_data() + k*moxy, u_2_2_i_mul_d_y_theta_j_fourier, C, theta_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "theta_4 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//...summand 5 of theta
            for(int k = 0; k < moz; ++k) {
                double C = coeff.Ia12(i,k,j);
                if(C != 0.0) {
                    add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(theta_return_data->get_data() + k*moxy, u_3_1_i_mul_theta_j_fourier, C, theta_return_data->get_data() + k*moxy, moxy);
                }
                #ifdef DEBUG
                    cout << "theta_4 (i,j,k)=("<< i <<","<< j <<","<< k <<") coeff="<< C << endl;
                #endif
            }

			//free temporary memory 
			cudaFree(u_1_1_i_mul_u_1_1_j_fourier);
			cudaFree(u_1_1_i_mul_u_2_1_j_fourier);
			cudaFree(u_1_1_i_mul_u_3_1_j_fourier);
			cudaFree(u_1_1_i_mul_u_1_2_j_fourier);
			cudaFree(u_1_1_i_mul_u_2_2_j_fourier);
			cudaFree(u_2_1_i_mul_u_2_1_j_fourier);
			cudaFree(u_2_1_i_mul_u_3_1_j_fourier);			
			cudaFree(u_2_1_i_mul_u_1_2_j_fourier);			
			cudaFree(u_2_1_i_mul_u_2_2_j_fourier);			
			cudaFree(u_3_1_i_mul_u_3_1_j_fourier);			
			cudaFree(u_3_1_i_mul_u_1_2_j_fourier);			
			cudaFree(u_3_1_i_mul_u_2_2_j_fourier);			
			cudaFree(u_1_2_i_mul_u_1_2_j_fourier);			
			cudaFree(u_1_2_i_mul_u_2_2_j_fourier);			
			cudaFree(u_2_2_i_mul_u_2_2_j_fourier);
			cudaFree(u_1_1_i_mul_d_x_theta_j_fourier);			
			cudaFree(u_1_2_i_mul_d_x_theta_j_fourier);
			cudaFree(u_2_1_i_mul_d_y_theta_j_fourier);
			cudaFree(u_2_2_i_mul_d_y_theta_j_fourier);
			cudaFree(u_3_1_i_mul_theta_j_fourier);

			#ifdef DEBUG
                DBGOUT("transform back to fourier space finished");
			#endif

        }

#ifdef PENALIZED
        //subtract the penalization parts
		#ifdef DEBUG
            cout << "penalized parts" << endl;
		#endif

        //calculate \hat{mask \cdot u_1_1_j} (q, t)
        CUDA_FLOAT* mask_mul_u_1_1_j_fourier;
        cudaMalloc((void**) &mask_mul_u_1_1_j_fourier, sizeof(CUDA_FLOAT) * moxy);
		//...mult pointwise
        mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(mask, u_1_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
		//...transform back to fourier space			
        CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, mask_mul_u_1_1_j_fourier);


        //calculate \hat{mask \cdot u_2_1_j} (q, t)
        CUDA_FLOAT* mask_mul_u_2_1_j_fourier;
        cudaMalloc((void**) &mask_mul_u_2_1_j_fourier, sizeof(CUDA_FLOAT) * moxy);
		//...mult pointwise
        mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(mask, u_2_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
		//...transform back to fourier space			
        CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, mask_mul_u_2_1_j_fourier);


        //calculate \hat{mask \cdot u_3_1_j} (q, t)
        CUDA_FLOAT* mask_mul_u_3_1_j_fourier;
        cudaMalloc((void**) &mask_mul_u_3_1_j_fourier, sizeof(CUDA_FLOAT) * moxy);
		//...mult pointwise
        mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(mask, u_3_1_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
		//...transform back to fourier space			
        CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, mask_mul_u_3_1_j_fourier);


        //calculate \hat{mask \cdot u_1_2_j} (q, t)
        CUDA_FLOAT* mask_mul_u_1_2_j_fourier;
        cudaMalloc((void**) &mask_mul_u_1_2_j_fourier, sizeof(CUDA_FLOAT) * moxy);
		//...mult pointwise
        mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(mask, u_1_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
		//...transform back to fourier space			
        CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, mask_mul_u_1_2_j_fourier);

        //calculate \hat{mask \cdot u_2_2_j} (q, t)
        CUDA_FLOAT* mask_mul_u_2_2_j_fourier;
        cudaMalloc((void**) &mask_mul_u_2_2_j_fourier, sizeof(CUDA_FLOAT) * moxy);
		//...mult pointwise
        mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(mask, u_2_2_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
		//...transform back to fourier space			
        CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, mask_mul_u_2_2_j_fourier);

		
        //calculate \hat{mask \cdot theta_j} (q, t)
        CUDA_FLOAT* mask_mul_theta_j_fourier;
        cudaMalloc((void**) &mask_mul_theta_j_fourier, sizeof(CUDA_FLOAT) * moxy);
		//...mult pointwise
        mult_pointwise_real_matrix<<<grid_f_real,block_f_real>>>(mask, theta_j, num_elements_real_buffer, fourier_r2c_scaling, num_elements_real);
		//...transform back to fourier space			
        CUFFT_EXEC_R2C(r2c_plan, num_elements_real_buffer, mask_mul_theta_j_fourier);




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for f//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
		#ifdef DEBUG
            DBGOUT("calculate penalized f");
        #endif

		//...summand 1
        create_derivative_x<<<grid_moxy,block_moxy>>>(mask_mul_u_1_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
        for(int k = 0; k < moz; ++k) {
            double C = coeff.Iac(j,k);
            if(C != 0.0) {
                add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, prandtl_number/eta * C, f_return_data->get_data() + k*moxy, moxy);
            }
            #ifdef DEBUG
                cout << "pen_f_1 (j,k)=("<< j <<","<< k <<") coeff="<< C << endl;
            #endif
        }


		//...summand 2
        create_derivative_x<<<grid_moxy,block_moxy>>>(mask_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
        for(int k = 0; k < moz; ++k) {
            double C = coeff.Ia2(k,j);
            if(C != 0.0) {
                add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, prandtl_number/eta * C, f_return_data->get_data() + k*moxy, moxy);
            }
            #ifdef DEBUG
                cout << "pen_f_2 (j,k)=("<< j <<","<< k <<") coeff="<< C << endl;
            #endif
        }


		//...summand 3
        create_derivative_y<<<grid_moxy,block_moxy>>>(mask_mul_u_2_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
        for(int k = 0; k < moz; ++k) {
            double C = coeff.Iac(j,k);
            if(C != 0.0) {
                add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, prandtl_number/eta * C, f_return_data->get_data() + k*moxy, moxy);
            }
            #ifdef DEBUG
                cout << "pen_f_3 (j,k)=("<< j <<","<< k <<") coeff="<< C << endl;
            #endif
        }


		//...summand 4
        create_derivative_y<<<grid_moxy,block_moxy>>>(mask_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
        for(int k = 0; k < moz; ++k) {
            double C = coeff.Ia2(k,j);
            if(C != 0.0) {
                add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, prandtl_number/eta * C, f_return_data->get_data() + k*moxy, moxy);
            }
            #ifdef DEBUG
                cout << "pen_f_4 (j,k)=("<< j <<","<< k <<") coeff="<< C << endl;
            #endif
        }


		//...summand 5
        create_laplace_xy<<<grid_moxy,block_moxy>>>(mask_mul_u_3_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x, cube_length_y);
        add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + j*moxy, moxy_complex_buffer, -1.0*prandtl_number/eta, f_return_data->get_data() + j*moxy, moxy);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for g//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		//...summand 1
        create_derivative_y<<<grid_moxy,block_moxy>>>(mask_mul_u_1_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
        for(int k = 0; k < moz; ++k) {
            double C = -coeff.Ia2(j,k);
            if(C != 0.0) {
                add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, prandtl_number/eta * C, f_return_data->get_data() + k*moxy, moxy);
            }
            #ifdef DEBUG
                cout << "pen_g_1 (j,k)=("<< j <<","<< k <<") coeff="<< C << endl;
            #endif
        }


		//...summand 2
        create_derivative_y<<<grid_moxy,block_moxy>>>(mask_mul_u_1_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
        add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + j*moxy, moxy_complex_buffer, 0.5*prandtl_number/eta, f_return_data->get_data() + j*moxy, moxy);


		//...summand 3
        create_derivative_x<<<grid_moxy,block_moxy>>>(mask_mul_u_2_1_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_x);
        for(int k = 0; k < moz; ++k) {
            double C = -coeff.Ia2(j,k);
            if(C != 0.0) {
                add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + k*moxy, moxy_complex_buffer, -1.0*prandtl_number/eta * C, f_return_data->get_data() + k*moxy, moxy);
            }
            #ifdef DEBUG
                cout << "pen_g_3 (j,k)=("<< j <<","<< k <<") coeff="<< C << endl;
            #endif
        }


		//...summand 4
        create_derivative_y<<<grid_moxy,block_moxy>>>(mask_mul_u_2_2_j_fourier, moxy_complex_buffer, mox, moy, 1, cube_length_y);
        add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(f_return_data->get_data() + j*moxy, moxy_complex_buffer, -0.5*prandtl_number/eta, f_return_data->get_data() + j*moxy, moxy);




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for F//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			
		#ifdef DEBUG
            DBGOUT("calculate penalized F");
		#endif
		
        //...summand 1 of F
        for(int k = 0; k < moz; ++k) {
            double C = -coeff.Ia2(j,k);
            if(C != 0.0) {
                addup_only_dc_part<<<grid_moz,block_moz>>>(F_return_data->get_data()+k, mask_mul_u_1_1_j_fourier, mox, moy, 1, prandtl_number/eta * C);
            }
            #ifdef DEBUG
                cout << "pen_F_1 (j,k)=("<< j <<","<< k <<") coeff="<< C << endl;
            #endif
        }


		//...summand 2 of F
        addup_only_dc_part<<<grid_moz,block_moz>>>(F_return_data->get_data()+j, mask_mul_u_1_2_j_fourier, mox, moy, 1, 0.5*prandtl_number/eta);



		
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for G//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			
		#ifdef DEBUG
            DBGOUT("calculate penalized G");
		#endif
		
        //...summand 1 of G
        for(int k = 0; k < moz; ++k) {
            double C = -coeff.Ia2(j,k);
            if(C != 0.0) {
                addup_only_dc_part<<<grid_moz,block_moz>>>(G_return_data->get_data()+k, mask_mul_u_2_1_j_fourier, mox, moy, 1, prandtl_number/eta * C);
            }
            #ifdef DEBUG
                cout << "pen_G_1 (j,k)=("<< j <<","<< k <<") coeff="<< C << endl;
            #endif
        }


		//...summand 2 of G
        addup_only_dc_part<<<grid_moz,block_moz>>>(G_return_data->get_data()+j, mask_mul_u_2_2_j_fourier, mox, moy, 1, 0.5*prandtl_number/eta);




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////calculate nonlinear part for theta//////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		#ifdef DEBUG
            DBGOUT("calculate penalized theta");
		#endif

		//...summand 1 of theta
        add_mult_pointwise_on_device<<<grid_moxy,block_moxy>>>(theta_return_data->get_data() + j*moxy, mask_mul_theta_j_fourier, 0.5/eta, theta_return_data->get_data() + j*moxy, moxy);

        //free memory
		cudaFree(mask_mul_u_1_1_j_fourier);
		cudaFree(mask_mul_u_2_1_j_fourier);
		cudaFree(mask_mul_u_3_1_j_fourier);
		cudaFree(mask_mul_u_1_2_j_fourier);
		cudaFree(mask_mul_u_2_2_j_fourier);
		cudaFree(mask_mul_theta_j_fourier);
#endif /* PENALIZED */
    }



    // Check device for errors
    //ifdef DEBUG
        DBGSYNC();
    //endif

	//free memory
	cudaFree(u_1_1);
	cudaFree(u_2_1);
	cudaFree(u_3_1);
	cudaFree(u_1_2);
    cudaFree(u_2_2);
	cudaFree(theta_physical);
	cudaFree(d_x_theta_physical);
	cudaFree(d_y_theta_physical);	
    cudaFree(num_elements_real_buffer);
    cudaFree(moxy_complex_buffer);

	//apply scaling factor to F and G
	theta_return_data->scale_itself((-1.0));
    f_return_data->scale_itself((-1.0)/prandtl_number);
    g_return_data->scale_itself((-1.0)/prandtl_number);
    F_return_data->scale_itself((-1.0)/prandtl_number);
    G_return_data->scale_itself((-1.0)/prandtl_number);

	//set return value
	matrix_folder* theta_return = new matrix_folder(1);
	matrix_folder* f_return = new matrix_folder(1);
	matrix_folder* g_return = new matrix_folder(1);
	matrix_folder* F_return = new matrix_folder(1);
	matrix_folder* G_return = new matrix_folder(1);
	theta_return->add_matrix(0, theta_return_data);
	f_return->add_matrix(0, f_return_data);
	g_return->add_matrix(0, g_return_data);
	F_return->add_matrix(0, F_return_data);
	G_return->add_matrix(0, G_return_data);
	return_array[0] = theta_return;
	return_array[1] = f_return;
	return_array[2] = g_return;
	return_array[3] = F_return;
	return_array[4] = G_return;

	#ifdef DEBUG
        DBGOUT("calculate nonlinear operator finished");
    #endif

	return return_array;
}


__host__ static dim3 create_block_dim(int number_of_matrix_entries){
    dim3 block;
    block.x = MAX_NUMBER_THREADS_PER_BLOCK;
    return block;
}


__host__ static dim3 create_grid_dim(int num){
    dim3 grid;
    // grid.x = ceil(num / MAX_N...)
    grid.x = (num + MAX_NUMBER_THREADS_PER_BLOCK - 1) / MAX_NUMBER_THREADS_PER_BLOCK;
    return grid;
}

__device__ static int get_global_index(){
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

// Differentiate matrix of fourier coefficients in x-direction
// output[kx,ky,kz] = input[kx,ky,kz] * 2pi*i * kx/cube_length_x
__global__ void create_derivative_x(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x) {
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices) {
        CUDA_FLOAT inval = input[total_index];

        int col_index = 0, row_index = 0, matrix_index = 0;
        get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, matrices);

        int k_x = col_index;

        // Derivative is ik
        CUDA_FLOAT_REAL fac = k_x * (2*M_PI / cube_length_x);

        CUDA_FLOAT outval;
        outval.x = -fac * inval.y;
        outval.y =  fac * inval.x;
        output[total_index] = outval;
    }
}

// Differentiate matrix of fourier coefficients in y-direction
// output[kx,ky,kz] = input[kx,ky,kz] * 2pi*i * ky/cube_length_y
__global__ void create_derivative_y(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_y) {
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices) {
        CUDA_FLOAT inval = input[total_index];

        int col_index = 0, row_index = 0, matrix_index = 0;
        get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, matrices);

        // k_y is wrapped at rows/2
        int k_y = row_index;
        if(k_y > rows/2)
            k_y -= rows;

        // Derivative is ik
        CUDA_FLOAT_REAL fac = k_y * (2*M_PI / cube_length_y);

        CUDA_FLOAT outval;
        outval.x = -fac * inval.y;
        outval.y =  fac * inval.x;
        output[total_index] = outval;
    }
}

// Differentiate matrix of fourier coefficients in xx-direction
// output[kx,ky,kz] = input[kx,ky,kz] * (2pi*i * kx/cube_length_x)^2
__global__ void create_second_derivative_xx(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x){
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices) {
        CUDA_FLOAT entry = input[total_index];

        int col_index = 0, row_index = 0, matrix_index = 0;
        get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, matrices);

        int k_x = col_index;

        // Derivative is -(2pi k/clx)^2
        CUDA_FLOAT_REAL fac = k_x * (2*M_PI / cube_length_x);
        fac = -fac*fac;

        entry.x *= fac;
        entry.y *= fac;
        output[total_index] = entry;
    }
}

// Differentiate matrix of fourier coefficients in yy-direction
// output[kx,ky,kz] = input[kx,ky,kz] * (2pi*i * ky/cube_length_y)^2
__global__ void create_second_derivative_yy(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_y){
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices) {
        CUDA_FLOAT entry = input[total_index];

        int col_index = 0, row_index = 0, matrix_index = 0;
        get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, matrices);

        // k_y is wrapped at rows/2
        int k_y = row_index;
        if(k_y > rows/2)
            k_y -= rows;

        // Derivative is -(2pi k/cly)^2
        CUDA_FLOAT_REAL fac = k_y * (2*M_PI / cube_length_y);
        fac = -fac*fac;

        entry.x *= fac;
        entry.y *= fac;
        output[total_index] = entry;
    }
}

// Differentiate matrix of fourier coefficients in xy-direction
// output[kx,ky,kz] = input[kx,ky,kz] * (2pi*i)^2 * (kx*ky)/(cube_length_x * cube_length_y)
__global__ void create_second_derivative_xy(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y){
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices) {
        CUDA_FLOAT entry = input[total_index];

        int col_index = 0, row_index = 0, matrix_index = 0;
        get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, matrices);

        // k_y is wrapped at rows/2
        int k_x = col_index;
        int k_y = row_index;
        if(k_y > rows/2)
            k_y -= rows;

        // Derivative is -(2pi kx/clx)*(2pi ky/cly)
        CUDA_FLOAT_REAL fac = -4*M_PI*M_PI * (k_x*k_y) / (cube_length_x*cube_length_y);

        entry.x *= fac;
        entry.y *= fac;
        output[total_index] = entry;
    }
}

// Apply laplace operator to matrix of fourier coefficients in xy-direction
// output[kx,ky,kz] = input[kx,ky,kz] * [(2pi*i * kx/cube_length_x)^2+(2pi*i * ky/cube_length_y)^2]
__global__ void create_laplace_xy(CUDA_FLOAT* input, CUDA_FLOAT* output, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y){
    int total_index = get_global_index();

    //check if thread is valid
    if(total_index < columns*rows*matrices) {
        CUDA_FLOAT entry = input[total_index];

        int col_index = 0, row_index = 0, matrix_index = 0;
        get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, matrices);

        // k_y is wrapped at rows/2
        int k_x = col_index;
        int k_y = row_index;
        if(k_y > rows/2)
            k_y -= rows;

        // Laplace is -[(2pi kx/clx)^2+(2pi ky/cly)^2]
        CUDA_FLOAT_REAL fac_x = k_x / cube_length_x;
        CUDA_FLOAT_REAL fac_y = k_y / cube_length_y;
        CUDA_FLOAT_REAL fac = (-4*M_PI*M_PI) * (fac_x * fac_x + fac_y * fac_y);

        entry.x *= fac;
        entry.y *= fac;
        output[total_index] = entry;
    }

}

__global__ void mult_pointwise_real_matrix(CUDA_FLOAT_REAL* input_1, CUDA_FLOAT_REAL* input_2, CUDA_FLOAT_REAL* output, CUDA_FLOAT_REAL factor, int num_entries){
    int total_index = get_global_index();

	//check if thread is valid
	if(total_index < num_entries) {
        output[total_index] = factor * input_1[total_index] * input_2[total_index];
	}
}

// This function adds F to the direct current (k=0) components of g
// g[index_xyz] += F[index_z]
__global__ void add_dc_part(CUDA_FLOAT* g, CUDA_FLOAT* F, int columns, int rows, int matrices) {
    int index_z = get_global_index();

    if(index_z < matrices) {
        int index_xyz = columns * rows * index_z;
        CUDA_FLOAT entry = g[index_xyz];
        CUDA_FLOAT add_entry = F[index_z];
        entry.x += add_entry.x;
        entry.y += add_entry.y;

        g[index_xyz] = entry;
    }
}

// This function adds the direct current part (k=0) component of fourier coefficients to F
// F[index_z] += factor * f[index_xyz]
__global__ void addup_only_dc_part(CUDA_FLOAT* F, CUDA_FLOAT* f, int f_columns, int f_rows, int f_matrices, CUDA_FLOAT_REAL factor) {

    int index_z = get_global_index();

    if(index_z < f_matrices) {
        CUDA_FLOAT entry = F[index_z];
        CUDA_FLOAT add_entry = f[f_columns * f_rows * index_z];
        entry.x += add_entry.x * factor;
        entry.y += add_entry.y * factor;

        F[index_z] = entry;
    }
}

// output[index] = factor * input[index]
// input == output is allowed
__global__ void scale_on_device(CUDA_FLOAT* input, CUDA_FLOAT_REAL factor, CUDA_FLOAT* output, int number_of_matrix_entries){
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        CUDA_FLOAT entry = input[index];

        entry.x *= factor;
        entry.y *= factor;
        output[index] = entry;
    }
}

// output[index] = input[index] + factor * to_add[index]
__global__ void add_mult_pointwise_on_device(CUDA_FLOAT* input, CUDA_FLOAT* to_add, CUDA_FLOAT_REAL factor, CUDA_FLOAT* output, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        CUDA_FLOAT entry = input[index];
        CUDA_FLOAT add_entry = to_add[index];

        entry.x += add_entry.x * factor;
        entry.y += add_entry.y * factor;
        output[index] = entry;
    }
}

#ifdef PENALIZED
__global__ static void init_mask_physical_space_rectangle(CUDA_FLOAT_REAL* mask,int columns,int rows){
    int total_index = get_global_index();
    int col_index = 0, row_index = 0, matrix_index = 0;
    get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, 1);

	//use a mask
	if(total_index < columns*rows){
		//init all with zeros
        CUDA_FLOAT_REAL mask_val = 0.0;

		//rectangular mask
        if((col_index < 3) || (col_index > columns-4)
                || (row_index < 3) || (row_index > rows-4)) {
            mask_val = 1.0;
        }

        mask[total_index] = mask_val;
	}
}

__global__ static void init_mask_physical_space_circle(CUDA_FLOAT_REAL* mask,int columns,int rows){
    int total_index = get_global_index();
    int col_index = 0, row_index = 0, matrix_index = 0;
    get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, 1);

	//use a mask
	if(total_index < columns*rows){
        //init all with zeros
        CUDA_FLOAT_REAL mask_val = 0.0;

		//rectangular mask
        int distance_from_center = (col_index - (columns/2))*(col_index - (columns/2)) + (row_index - (rows/2))*(row_index - (rows/2));

		int min = columns; 
		if(rows < columns)
			min = rows;
        if(distance_from_center > (min/2-3) * (min/2-3) )
            mask_val = 1.0;

        mask[total_index] = mask_val;
	}
}

__global__ static void init_mask_physical_space_disk_array(CUDA_FLOAT_REAL* mask, int columns, int rows, int number_of_disks_x, int number_of_disks_y){
    int total_index = get_global_index();
    int col_index = 0, row_index = 0, matrix_index = 0;
    get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, 1);

	//use a mask
	if(total_index < columns*rows){
        //init all with zeros
        CUDA_FLOAT_REAL mask_val = 0.0;

		//check distance form all centers
		//...loop over all centers in x direction
		for(int i = 0; i < number_of_disks_x; i++) {
			//...loop over all centers in y direction
			for(int j = 0; j < number_of_disks_y; j++){
				
				int center_x_index = i * columns/number_of_disks_x  + columns/(number_of_disks_x*2);
				int center_y_index = j * rows/number_of_disks_y + rows/(number_of_disks_y*2);
				
				//calc distance from the center
				int distance_squared = (center_x_index - col_index)*(center_x_index - col_index) + (center_y_index - row_index)*(center_y_index - row_index); 
				
				//calc the radius of the disks
				int min_squared_distance = columns/(number_of_disks_x*2);
				if(min_squared_distance > rows/(number_of_disks_y*2))
					min_squared_distance = rows/(number_of_disks_y*2);
                if(distance_squared < min_squared_distance)
                    mask_val = 1.0;
			}
        }

        mask[total_index] = mask_val;
	}
}

__global__ static void init_mask_physical_space_everywhere(CUDA_FLOAT_REAL* mask, int columns, int rows){
    int total_index = get_global_index();

    //use a mask
    if(total_index < columns * rows) {
        //init all with zeros
        mask[total_index] = 1.;
    }
}
#endif /* PENALIZED */



