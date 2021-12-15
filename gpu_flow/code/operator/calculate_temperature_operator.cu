#include "calculate_temperature_operator.h"


calculate_temperature_operator* calculate_temperature_operator::init(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length){
	calculate_temperature_operator* op = new calculate_temperature_operator(dimension, cube_length);
	return op;	
}


calculate_temperature_operator::~calculate_temperature_operator(){
	cufftDestroy(c2r_plan);
}

// returns a vector-valued temperature field
// num_z must be at least 2 for the algorithm to work, and at least 3 for reasonable results
matrix_folder_real *calculate_temperature_operator::calculate_operator(matrix_folder* theta, const int num_z) {

    // Define z-positions: linspace, including values at -0.5 and 0.5 (those are zero)
    vector<CUDA_FLOAT_REAL> at_z(num_z);
    for(int k = 0; k < num_z; k++) {
        at_z.at(k) = (k * cube_length_z) / ( (CUDA_FLOAT_REAL) (num_z-1) ) - 0.5*cube_length_z;
    }

    return calculate_operator_at(theta, at_z);
}

matrix_folder_real* calculate_temperature_operator::calculate_operator_at(matrix_folder* theta, const vector<CUDA_FLOAT_REAL>& at_z) {

    std::vector<int> theta_dim = theta->get_matrix(0)->get_matrix_dimension();
    const int num_x_real = 2*(theta_dim.at(0) -1);
    const int num_y_real = theta_dim.at(1);
    const int num_z_real = at_z.size();
    const int num_xy_real = num_x_real * num_y_real;


    //...build theta(x, y, n, t)
    CUDA_FLOAT* theta_data = theta->get_matrix(0)->get_data();
	CUDA_FLOAT_REAL* theta_real;
    cudaMalloc((void**) &theta_real, sizeof(CUDA_FLOAT_REAL) * theta_dim.at(2) * num_xy_real);
    for(int i = 0; i < theta_dim.at(2); i++) {
        CUFFT_EXEC_C2R(c2r_plan, theta_data+i*(theta_dim.at(0)*theta_dim.at(1)) , theta_real + i*num_xy_real);
	}


    // Create return matrix and set to zero
    std::vector<int> real_dimensions(3);
    real_dimensions[0] = num_x_real;
    real_dimensions[1] = num_y_real;
    real_dimensions[2] = num_z_real;
    matrix_device_real* theta_return = new matrix_device_real(real_dimensions);
    theta_return->set_zero();

    // Sample temperature field in z-direction
    dim3 grid_dim = create_grid_dim(num_xy_real);
    dim3 block_dim = create_block_dim(num_xy_real);
    for(int k = 0; k < num_z_real; k++) {

		//current evaluation point
        CUDA_FLOAT_REAL z = at_z.at(k); // z in [-0.5:0.5]
		
        // Sum up contributions from vertical ansatz functions
		for(int i = 0; i < theta_dim.at(2); i++) {
		
			//...factors
            CUDA_FLOAT_REAL S_i = sin(M_PI*(i+1)*(z+0.5)); 	// S_i(z = z_i)

            //build temperature field
            mult_add_pointwise<<<grid_dim,block_dim>>>(theta_real+i*num_xy_real, S_i, num_xy_real, theta_return->get_data()+k*num_xy_real);
		}
    }

	//free memory
    cudaFree(theta_real);

	//return folder with temperature distribution
    matrix_folder_real* t = new matrix_folder_real(1);
	t->add_matrix(0, theta_return);
	return t;
}

calculate_temperature_operator::calculate_temperature_operator(std::vector<int> dimension, std::vector<CUDA_FLOAT_REAL> cube_length){

	//get length of the cube
	cube_length_x = cube_length.at(0);
	cube_length_y = cube_length.at(1);
    cube_length_z = cube_length.at(2);

	//create fft plan
    int num_x = 2*(dimension.at(0)-1);
	int num_y = dimension.at(1);
	cufftPlan2d(&c2r_plan,num_y,num_x,CUFFT_C2R);
}



__host__ dim3 create_block_dim(int number_of_matrix_entries){
    dim3 block;
    block.x = MAX_NUMBER_THREADS_PER_BLOCK;
    return block;
}


__host__ dim3 create_grid_dim(int number_of_matrix_entries){
    dim3 grid;
    grid.x = (number_of_matrix_entries + MAX_NUMBER_THREADS_PER_BLOCK - 1) / MAX_NUMBER_THREADS_PER_BLOCK;

    return grid;
}

__device__ static int get_global_index() {
    return (threadIdx.x + (threadIdx.y + (threadIdx.z + (blockIdx.x + (blockIdx.y + (blockIdx.z)
            * gridDim.y) * gridDim.x) * blockDim.z) * blockDim.y) * blockDim.x);
}

__global__ void mult_add_pointwise(CUDA_FLOAT_REAL* input_1, CUDA_FLOAT_REAL factor_1, int number_of_matrix_entries, CUDA_FLOAT_REAL* input_and_output) {
    int index = get_global_index();
	if(index < number_of_matrix_entries) {
        input_and_output[index] += factor_1 * input_1[index];
	}
}





