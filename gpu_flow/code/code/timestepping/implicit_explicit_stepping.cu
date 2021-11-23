#include "implicit_explicit_stepping.h"

implicit_explicit_step::implicit_explicit_step(linear_implicit_operator* impl_operator, inverse_B_matrix* inv_B, nonlinear_operator_rayleigh_noslip* nonlin_op, CUDA_FLOAT_REAL delta_t){
    implicit_op = impl_operator;
    inverse_B_op = inv_B;
    nonlinear_operator = nonlin_op;
    res_1 = NULL;
}
	
	
implicit_explicit_step::~implicit_explicit_step(){
    delete implicit_op;
    delete inverse_B_op;
    delete nonlinear_operator;


	for(int i = 0; i < 5; i++){
        if(res_1 != NULL && res_1[i] != NULL)
            delete res_1[i];
	}
    if(res_1 != NULL)
        delete res_1;
}


implicit_explicit_step* implicit_explicit_step::init_timestepping(linear_implicit_operator* impl_operator, inverse_B_matrix* inv_B, nonlinear_operator_rayleigh_noslip* nonlin_op, CUDA_FLOAT_REAL delta_t){
	implicit_explicit_step* op = new implicit_explicit_step(impl_operator, inv_B, nonlin_op, delta_t);
	return op;
}


void implicit_explicit_step::step_time(matrix_folder** input_and_output, CUDA_FLOAT_REAL delta_t){

	//first calculate non-linear part
	#ifdef DEBUG
        DBGOUT("step time with adams bashforth method for nonlinear part");
	#endif

	//...dealias data before calculating the nonlinear part!
    dealias_data(input_and_output);

    int begin_index = 0;
    if(res_1 == NULL){
		//... a simple euler step
        matrix_folder** tmp = nonlinear_operator->calculate_operator(input_and_output[0],input_and_output[1],input_and_output[2], input_and_output[3], input_and_output[4]);
        res_1 = inverse_B_op->calculate_inverse(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4]);
		for(int i = 0; i< 5; i++)
            delete tmp[i];

		for(int i = begin_index; i < 5; i++){
			input_and_output[i]->add_mult_itself_pointwise(res_1[i], delta_t);
		}
    } else {
		//adams bashforth of order 2
        matrix_folder** tmp = nonlinear_operator->calculate_operator(input_and_output[0], input_and_output[1], input_and_output[2], input_and_output[3], input_and_output[4]);
        matrix_folder** current = inverse_B_op->calculate_inverse(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]);
		for(int i = 0; i< 5; i++)
			delete tmp[i];
        for(int i = begin_index; i < 5; i++) {
            input_and_output[i]->add_mult_itself_pointwise(current[i], 1.5 * delta_t);
            input_and_output[i]->add_mult_itself_pointwise(res_1[i], -0.5 * delta_t);
		}

        //free memory and rearrange previous data
        for(int i = 0; i < 5; i++) delete res_1[i];
        delete res_1;
        res_1 = current;
    }

	//...dealias data before calculating the linear part and after calculating the nonlinear part!
    dealias_data(input_and_output);

	#ifdef DEBUG
        DBGOUT("step time with adams bashforth method for nonlinear part finished");
	#endif


	//now calculate linear implicit part
	#ifdef DEBUG
        DBGOUT("step time with implicit euler method for linear part");
    #endif

    matrix_folder** linear_output = implicit_op->calculate_operator(input_and_output[0], input_and_output[1], input_and_output[2], input_and_output[3], input_and_output[4]);
	//copy the data to output array
	for(int i = 0; i < 5; i++)
		copy_data_to_folder(linear_output[i], input_and_output[i]);
	for(int i = 0; i < 5; i++)
        delete linear_output[i];

	
    #ifdef DEBUG
        cudaError_t cudaError = cudaGetLastError();
        if( cudaError != cudaSuccess ) {
            EXIT_ERROR2("CUDA Runtime API Error reported (after implicit operator) : ", cudaGetErrorString(cudaError));
		}
	#endif
}

void implicit_explicit_step::copy_data_to_folder(matrix_folder* src, matrix_folder* target) {
	//iterate over the number of vector components
	int vector_comp = src->get_dimension();

	//copy the data for each vector component
	for(int i = 0; i < vector_comp; i++) {
		CUDA_FLOAT* src_data = src->get_matrix(i)->get_data();
		CUDA_FLOAT* target_data = target->get_matrix(i)->get_data();

		int columns = src->get_matrix(i)->get_matrix_dimension().at(0);
		int rows = src->get_matrix(i)->get_matrix_dimension().at(1);
		int matrices = src->get_matrix(i)->get_matrix_dimension().at(2);

		dim3 grid_dim = src->get_matrix(i)->create_grid_dimension();
		dim3 block_dim = src->get_matrix(i)->create_block_dimension();

		copy_data<<< grid_dim , block_dim >>>(src_data, target_data, columns, rows, matrices);
	}
}


void implicit_explicit_step::dealias_data(matrix_folder** data_vector){

	//only dealias theta, f and g	
	matrix_folder* theta = data_vector[0];
	matrix_folder* f = data_vector[1];
	matrix_folder* g = data_vector[2];

	//...dealias theta
	matrix_device* theta_matrix = theta->get_matrix(0);
	dim3 theta_grid = theta_matrix->create_grid_dimension();
	dim3 theta_block = theta_matrix->create_block_dimension();
    dealias_implicit<<< theta_grid, theta_block >>>(theta_matrix->get_data(), theta_matrix->get_matrix_size(0), theta_matrix->get_matrix_size(1), theta_matrix->get_matrix_size(2));


	//...dealias f
	matrix_device* f_matrix = f->get_matrix(0);
	dim3 f_grid = f_matrix->create_grid_dimension();
	dim3 f_block = f_matrix->create_block_dimension();
    dealias_implicit<<< f_grid, f_block >>>(f_matrix->get_data(), f_matrix->get_matrix_size(0), f_matrix->get_matrix_size(1), f_matrix->get_matrix_size(2));

	//...dealias g
	matrix_device* g_matrix = g->get_matrix(0);
	dim3 g_grid = g_matrix->create_grid_dimension();
	dim3 g_block = g_matrix->create_block_dimension();
    dealias_implicit<<< g_grid, g_block >>>(g_matrix->get_data(), g_matrix->get_matrix_size(0), g_matrix->get_matrix_size(1), g_matrix->get_matrix_size(2));
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

__global__ void dealias_implicit(CUDA_FLOAT* data, int columns, int rows, int matrices){

    int total_index = get_global_index();
    int current_col = 0, current_row = 0, current_matrix = 0;
    get_current_matrix_indices(current_col, current_row, current_matrix, total_index, columns, rows, matrices);

	//modes in x direction
	if(total_index < columns*rows*matrices) {
		if(columns - current_col <= (1.0/3.0)*columns) {
			data[total_index].x = 0.0;
			data[total_index].y = 0.0;
		}
		//modes in y direction
		if(fabs( 0.5*rows - current_row ) <= (1.0/3.0) * 0.5*rows){
			data[total_index].x = 0.0;
			data[total_index].y = 0.0;
		}
	
	}
}

__global__ void copy_data(CUDA_FLOAT* src_data, CUDA_FLOAT* target_data, int columns, int rows, int matrices){
    int total_index = get_global_index();

	if(total_index < columns*rows*matrices){
		target_data[total_index].x = src_data[total_index].x;
		target_data[total_index].y = src_data[total_index].y;
	}
}



