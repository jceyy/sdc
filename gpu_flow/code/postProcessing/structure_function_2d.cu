#include "structure_function_2d.h"


structure_function_2d::structure_function_2d(std::vector<int> dimensions, std::vector<CUDA_FLOAT_REAL> cube_length){
	this->dim_x = dimensions.at(0);
	this->dim_y = dimensions.at(1);
	this->dim_z = dimensions.at(2);

	this->cube_length_x = cube_length.at(0);
	this->cube_length_y = cube_length.at(1);
	this->cube_length_z = cube_length.at(2);

	//create fft plan
    //int num_x_physical = 2*(dimensions.at(0) -1);
    //int num_y_physical = dimensions.at(1);
	//cufftPlan2d(&c2r_plan,num_y_physical,num_x_physical,CUFFT_C2R);
	//...create inverse fft plan
	//cufftPlan2d(&r2c_plan,num_y_physical,num_x_physical,CUFFT_R2C);

}


structure_function_2d::~structure_function_2d(){

	//free cuda fft plans
	//cufftDestroy(c2r_plan);
	//cufftDestroy(r2c_plan);

}


matrix_folder* structure_function_2d::calculate_u_1_structure_function_x(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples){

	//create the right output
	matrix_folder* struct_function = new matrix_folder(1);
	std::vector<int> dimen; dimen.push_back(number_of_samples); dimen.push_back(1);
	matrix_device* struct_f_matrix = new matrix_device(dimen);

	//loop over different radius
	CUDA_FLOAT_REAL r_1 = min_r_radius;
	CUDA_FLOAT_REAL r_2 = 0.0;
	CUDA_FLOAT_REAL r_1_increment = ( max_r_radius - min_r_radius ) / number_of_samples;
	for(int i = 0 ; i < number_of_samples; i++){
		r_1 = min_r_radius + i * r_1_increment;
		//use the right ouput field
		calculate_structure_function_u_1(f,g, r_1, r_2, 2, struct_f_matrix->get_data() + i);
	}

	struct_function->add_matrix(0, struct_f_matrix);
	return struct_function;

}

matrix_folder* structure_function_2d::calculate_u_1_structure_function_y(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples){

	//create the right output
	matrix_folder* struct_function = new matrix_folder(1);
	std::vector<int> dimen; dimen.push_back(number_of_samples); dimen.push_back(1);
	matrix_device* struct_f_matrix = new matrix_device(dimen);

	//loop over different radius
	CUDA_FLOAT_REAL r_1 = 0.0;
	CUDA_FLOAT_REAL r_2 = min_r_radius;
	CUDA_FLOAT_REAL r_2_increment = ( max_r_radius - min_r_radius ) / number_of_samples;
	for(int i = 0 ; i < number_of_samples; i++){
		r_2 = min_r_radius + i * r_2_increment;
		//use the right ouput field
		calculate_structure_function_u_1(f,g, r_1, r_2, 2, struct_f_matrix->get_data() + i);
	}

	struct_function->add_matrix(0, struct_f_matrix);
	return struct_function;

}

matrix_folder* structure_function_2d::calculate_u_2_structure_function_x(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples){

	//create the right output
	matrix_folder* struct_function = new matrix_folder(1);
	std::vector<int> dimen; dimen.push_back(number_of_samples); dimen.push_back(1);
	matrix_device* struct_f_matrix = new matrix_device(dimen);

	//loop over different radius
	CUDA_FLOAT_REAL r_1 = min_r_radius;
	CUDA_FLOAT_REAL r_2 = 0.0;
	CUDA_FLOAT_REAL r_1_increment = ( max_r_radius - min_r_radius ) / number_of_samples;
	for(int i = 0 ; i < number_of_samples; i++){
		r_1 = min_r_radius + i * r_1_increment;
		//use the right ouput field
		calculate_structure_function_u_2(f,g, r_1, r_2, 2, struct_f_matrix->get_data() + i);
	}

	struct_function->add_matrix(0, struct_f_matrix);
	return struct_function;

}

matrix_folder* structure_function_2d::calculate_u_2_structure_function_y(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples){

	//create the right output
	matrix_folder* struct_function = new matrix_folder(1);
	std::vector<int> dimen; dimen.push_back(number_of_samples); dimen.push_back(1);
	matrix_device* struct_f_matrix = new matrix_device(dimen);

	//loop over different radius
	CUDA_FLOAT_REAL r_1 = 0.0;
	CUDA_FLOAT_REAL r_2 = min_r_radius;
	CUDA_FLOAT_REAL r_2_increment = ( max_r_radius - min_r_radius ) / number_of_samples;
	for(int i = 0 ; i < number_of_samples; i++){
		r_2 = min_r_radius + i * r_2_increment;
		//use the right ouput field
		calculate_structure_function_u_2(f,g, r_1, r_2, 2, struct_f_matrix->get_data() + i);
	}

	struct_function->add_matrix(0, struct_f_matrix);
	return struct_function;

}


void structure_function_2d::calculate_structure_function_u_2(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, int order, CUDA_FLOAT* output){
	//the current displacement
	CUDA_FLOAT_REAL r_x = r_1;
	CUDA_FLOAT_REAL r_y = r_2;

	//apply twiddle by shift, transform, .....
	matrix_device* t = f->get_matrix(0);
	matrix_device* t_2 = g->get_matrix(0);
	int columns = t->get_matrix_dimension().at(0);
	int rows = t->get_matrix_dimension().at(1);
	int matrices = t->get_matrix_dimension().at(2);
	std::vector<int> dim; dim.push_back(columns); dim.push_back(rows);
	matrix_device* fourier_coeff_structure_function = new matrix_device(dim); 
	matrix_device* fourier_coeff_structure_function_part_2 = new matrix_device(dim); 

	//calc in fourier space: u_1(x+r_x,y+r_y, 0)-u_1(x,y, 0)
	twiddle_by_shift_u_2<<< t->create_grid_dimension() , t->create_block_dimension() >>>(t->get_data(), t_2->get_data(), columns, rows, matrices, r_x, r_y, fourier_coeff_structure_function->get_data(), cube_length_x, cube_length_y, cube_length_z);
	twiddle_by_shift_u_2<<< t->create_grid_dimension() , t->create_block_dimension() >>>(t->get_data(), t_2->get_data(), columns, rows, matrices, r_x, r_y, fourier_coeff_structure_function_part_2->get_data(), cube_length_x, cube_length_y, cube_length_z);

	//calc in physical space: | T(x+r_x,y+r_y, 0)-T(x,y, 0) |^{ \frac{order}{2} }
	int num_x_physical = 2*(columns -1);
	int num_y_physical = rows;
	dim3 block_dim = create_block_dimension_real_matrix_structure_function(num_x_physical * num_y_physical);
	dim3 grid_dim = create_grid_dimension_real_matrix_structure_function(num_x_physical * num_y_physical);
	//...create FFTs
	cufftHandle c2r_plan;
	cufftHandle r2c_plan;
	cufftPlan2d(&c2r_plan,num_y_physical,num_x_physical,CUFFT_C2R);
	cufftPlan2d(&r2c_plan,num_y_physical,num_x_physical,CUFFT_R2C);
	//...with exponent = order/2
	CUDA_FLOAT_REAL* physical_part_1;
	cudaMalloc((void**) &physical_part_1, sizeof(CUDA_FLOAT_REAL) * num_x_physical * num_y_physical);
    CUFFT_EXEC_C2R(c2r_plan, fourier_coeff_structure_function->get_data() , physical_part_1);
	abs_and_square<<< grid_dim, block_dim >>>(physical_part_1, num_x_physical * num_y_physical, ((CUDA_FLOAT_REAL) order)/2.0);
	//...with exponent = order/2 + order%2
	CUDA_FLOAT_REAL* physical_part_2;
	cudaMalloc((void**) &physical_part_2, sizeof(CUDA_FLOAT_REAL) * num_x_physical * num_y_physical);
    CUFFT_EXEC_C2R(c2r_plan, fourier_coeff_structure_function_part_2->get_data() , physical_part_2);
	abs_and_square<<< grid_dim, block_dim >>>(physical_part_2, num_x_physical * num_y_physical, ((CUDA_FLOAT_REAL) order)/2.0);


	//transform to fourier space: | T(x+r_x,y+r_y, 0)-T(x,y, 0) |^{ \frac{order}{2} }
	matrix_device* part_1_fourier = new matrix_device(dim); 
	matrix_device* part_2_fourier = new matrix_device(dim); 
    CUFFT_EXEC_R2C(r2c_plan, physical_part_1 , part_1_fourier->get_data());
    CUFFT_EXEC_R2C(r2c_plan, physical_part_2 , part_2_fourier->get_data());
	//...rescale the fourier coefficients to get the right values
	scale_on_device<<< part_1_fourier->create_grid_dimension() , part_1_fourier->create_block_dimension() >>>(part_1_fourier->get_data(),1.0/((CUDA_FLOAT_REAL) num_x_physical*num_y_physical) , part_1_fourier->get_data(), columns*rows);
	scale_on_device<<< part_2_fourier->create_grid_dimension() , part_2_fourier->create_block_dimension() >>>(part_2_fourier->get_data(),1.0/((CUDA_FLOAT_REAL) num_x_physical*num_y_physical) , part_2_fourier->get_data(), columns*rows);
	cufftDestroy(c2r_plan);
	cufftDestroy(r2c_plan);

	//build scalar product of them
	int fourier_col = part_1_fourier->get_matrix_dimension().at(0);
	int fourier_row = part_1_fourier->get_matrix_dimension().at(1);
	mult_conj_pointwise_on_device<<<part_1_fourier->create_grid_dimension() , part_1_fourier->create_block_dimension()>>>(part_1_fourier->get_data(),part_2_fourier->get_data() , part_1_fourier->get_data(), fourier_col*fourier_row);
	//...now reduce part_1_fourier
	reduce(part_1_fourier->get_data(), fourier_col * fourier_row, output );

	//free memory
	delete fourier_coeff_structure_function;
	delete fourier_coeff_structure_function_part_2;
	cudaFree(physical_part_1);
	cudaFree(physical_part_2);
	delete part_1_fourier;
	delete part_2_fourier;

}


void structure_function_2d::calculate_structure_function_u_1(matrix_folder* f, matrix_folder* g, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, int order, CUDA_FLOAT* output){
	//the current displacement
	CUDA_FLOAT_REAL r_x = r_1;
	CUDA_FLOAT_REAL r_y = r_2;

	//apply twiddle by shift, transform, .....
	matrix_device* t = f->get_matrix(0);
	matrix_device* t_2 = g->get_matrix(0);
	int columns = t->get_matrix_dimension().at(0);
	int rows = t->get_matrix_dimension().at(1);
	int matrices = t->get_matrix_dimension().at(2);
	std::vector<int> dim; dim.push_back(columns); dim.push_back(rows);
	matrix_device* fourier_coeff_structure_function = new matrix_device(dim); 
	matrix_device* fourier_coeff_structure_function_part_2 = new matrix_device(dim); 

	//calc in fourier space: u_1(x+r_x,y+r_y, 0)-u_1(x,y, 0)
	twiddle_by_shift_u_1<<< t->create_grid_dimension() , t->create_block_dimension() >>>(t->get_data(), t_2->get_data(), columns, rows, matrices, r_x, r_y, fourier_coeff_structure_function->get_data(), cube_length_x, cube_length_y, cube_length_z);
	twiddle_by_shift_u_1<<< t->create_grid_dimension() , t->create_block_dimension() >>>(t->get_data(), t_2->get_data(), columns, rows, matrices, r_x, r_y, fourier_coeff_structure_function_part_2->get_data(), cube_length_x, cube_length_y, cube_length_z);

	//calc in physical space: | T(x+r_x,y+r_y, 0)-T(x,y, 0) |^{ \frac{order}{2} }
	int num_x_physical = 2*(columns -1);
	int num_y_physical = rows;
	dim3 block_dim = create_block_dimension_real_matrix_structure_function(num_x_physical * num_y_physical);
	dim3 grid_dim = create_grid_dimension_real_matrix_structure_function(num_x_physical * num_y_physical);
	//...create FFTs
	cufftHandle c2r_plan;
	cufftHandle r2c_plan;
	cufftPlan2d(&c2r_plan,num_y_physical,num_x_physical,CUFFT_C2R);
	cufftPlan2d(&r2c_plan,num_y_physical,num_x_physical,CUFFT_R2C);
	//...with exponent = order/2
	CUDA_FLOAT_REAL* physical_part_1;
	cudaMalloc((void**) &physical_part_1, sizeof(CUDA_FLOAT_REAL) * num_x_physical * num_y_physical);
    CUFFT_EXEC_C2R(c2r_plan, fourier_coeff_structure_function->get_data() , physical_part_1);
	abs_and_square<<< grid_dim, block_dim >>>(physical_part_1, num_x_physical * num_y_physical, ((CUDA_FLOAT_REAL) order)/2.0);
	//...with exponent = order/2 + order%2
	CUDA_FLOAT_REAL* physical_part_2;
	cudaMalloc((void**) &physical_part_2, sizeof(CUDA_FLOAT_REAL) * num_x_physical * num_y_physical);
    CUFFT_EXEC_C2R(c2r_plan, fourier_coeff_structure_function_part_2->get_data() , physical_part_2);
	abs_and_square<<< grid_dim, block_dim >>>(physical_part_2, num_x_physical * num_y_physical, ((CUDA_FLOAT_REAL) order)/2.0);


	//transform to fourier space: | T(x+r_x,y+r_y, 0)-T(x,y, 0) |^{ \frac{order}{2} }
	matrix_device* part_1_fourier = new matrix_device(dim); 
	matrix_device* part_2_fourier = new matrix_device(dim); 
    CUFFT_EXEC_R2C(r2c_plan, physical_part_1 , part_1_fourier->get_data());
    CUFFT_EXEC_R2C(r2c_plan, physical_part_2 , part_2_fourier->get_data());
	//...rescale the fourier coefficients to get the right values
	scale_on_device<<< part_1_fourier->create_grid_dimension() , part_1_fourier->create_block_dimension() >>>(part_1_fourier->get_data(),1.0/((CUDA_FLOAT_REAL) num_x_physical*num_y_physical) , part_1_fourier->get_data(), columns*rows);
	scale_on_device<<< part_2_fourier->create_grid_dimension() , part_2_fourier->create_block_dimension() >>>(part_2_fourier->get_data(),1.0/((CUDA_FLOAT_REAL) num_x_physical*num_y_physical) , part_2_fourier->get_data(), columns*rows);
	cufftDestroy(r2c_plan);
	cufftDestroy(c2r_plan);

	//build scalar product of them
	int fourier_col = part_1_fourier->get_matrix_dimension().at(0);
	int fourier_row = part_1_fourier->get_matrix_dimension().at(1);
	mult_conj_pointwise_on_device<<<part_1_fourier->create_grid_dimension() , part_1_fourier->create_block_dimension()>>>(part_1_fourier->get_data(),part_2_fourier->get_data() , part_1_fourier->get_data(), fourier_col*fourier_row);
	//...now reduce part_1_fourier
	reduce(part_1_fourier->get_data(), fourier_col * fourier_row, output );

	//free memory
	delete fourier_coeff_structure_function;
	delete fourier_coeff_structure_function_part_2;
	cudaFree(physical_part_1);
	cudaFree(physical_part_2);
	delete part_1_fourier;
	delete part_2_fourier;

}


matrix_folder* structure_function_2d::calculate_temperature_structure_function_x(matrix_folder* theta, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples){

	

	//create the right output
	matrix_folder* struct_function = new matrix_folder(1);
	std::vector<int> dimen; dimen.push_back(number_of_samples); dimen.push_back(1);
	matrix_device* struct_f_matrix = new matrix_device(dimen);

	//loop over different radius
	CUDA_FLOAT_REAL r_1 = min_r_radius;
	CUDA_FLOAT_REAL r_2 = 0.0;
	CUDA_FLOAT_REAL r_1_increment = ( max_r_radius - min_r_radius ) / number_of_samples;
	for(int i = 0 ; i < number_of_samples; i++){
		r_1 = min_r_radius + i * r_1_increment;
		//use the right ouput field
		calculate_structure_function(theta, r_1, r_2, 2, struct_f_matrix->get_data() + i);
	}

	struct_function->add_matrix(0, struct_f_matrix);
	return struct_function;
}

matrix_folder* structure_function_2d::calculate_temperature_structure_function_y(matrix_folder* theta, CUDA_FLOAT_REAL min_r_radius, CUDA_FLOAT_REAL max_r_radius, int number_of_samples){

	

	//create the right output
	matrix_folder* struct_function = new matrix_folder(1);
	std::vector<int> dimen; dimen.push_back(number_of_samples); dimen.push_back(1);
	matrix_device* struct_f_matrix = new matrix_device(dimen);

	//loop over different radius
	CUDA_FLOAT_REAL r_1 = 0.0;
	CUDA_FLOAT_REAL r_2 = min_r_radius;
	CUDA_FLOAT_REAL r_2_increment = ( max_r_radius - min_r_radius ) / number_of_samples;
	for(int i = 0 ; i < number_of_samples; i++){
		r_2 = min_r_radius + i * r_2_increment;
		//use the right ouput field
		calculate_structure_function(theta, r_1, r_2, 2, struct_f_matrix->get_data() + i);
	}

	struct_function->add_matrix(0, struct_f_matrix);
	return struct_function;
}


void structure_function_2d::calculate_structure_function(matrix_folder* theta, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, int order, CUDA_FLOAT* output){
	//the current displacement
	CUDA_FLOAT_REAL r_x = r_1;
	CUDA_FLOAT_REAL r_y = r_2;

	//apply twiddle by shift, transform, .....
	matrix_device* t = theta->get_matrix(0);
	int columns = t->get_matrix_dimension().at(0);
	int rows = t->get_matrix_dimension().at(1);
	int matrices = t->get_matrix_dimension().at(2);
	std::vector<int> dim; dim.push_back(columns); dim.push_back(rows);
	matrix_device* fourier_coeff_structure_function = new matrix_device(dim); 
	matrix_device* fourier_coeff_structure_function_part_2 = new matrix_device(dim); 

	//calc in fourier space: T(x+r_x,y+r_y, 0)-T(x,y, 0)
	twiddle_by_shift_temperature<<< t->create_grid_dimension() , t->create_block_dimension() >>>(t->get_data(), columns, rows, matrices, r_x, r_y, fourier_coeff_structure_function->get_data(), cube_length_x, cube_length_y, cube_length_z);
	twiddle_by_shift_temperature<<< t->create_grid_dimension() , t->create_block_dimension() >>>(t->get_data(), columns, rows, matrices, r_x, r_y, fourier_coeff_structure_function_part_2->get_data(), cube_length_x, cube_length_y, cube_length_z);

	//calc in physical space: | T(x+r_x,y+r_y, 0)-T(x,y, 0) |^{ \frac{order}{2} }
	int num_x_physical = 2*(columns -1);
	int num_y_physical = rows;
	dim3 block_dim = create_block_dimension_real_matrix_structure_function(num_x_physical * num_y_physical);
	dim3 grid_dim = create_grid_dimension_real_matrix_structure_function(num_x_physical * num_y_physical);
	//...create FFTs
	cufftHandle c2r_plan;
	cufftHandle r2c_plan;
	cufftPlan2d(&c2r_plan,num_y_physical,num_x_physical,CUFFT_C2R);
	cufftPlan2d(&r2c_plan,num_y_physical,num_x_physical,CUFFT_R2C);
	//...with exponent = order/2
	CUDA_FLOAT_REAL* physical_part_1;
	cudaMalloc((void**) &physical_part_1, sizeof(CUDA_FLOAT_REAL) * num_x_physical * num_y_physical);
    CUFFT_EXEC_C2R(c2r_plan, fourier_coeff_structure_function->get_data() , physical_part_1);
	abs_and_square<<< grid_dim, block_dim >>>(physical_part_1, num_x_physical * num_y_physical, ((CUDA_FLOAT_REAL) order)/2.0);
	//...with exponent = order/2 + order%2
	CUDA_FLOAT_REAL* physical_part_2;
	cudaMalloc((void**) &physical_part_2, sizeof(CUDA_FLOAT_REAL) * num_x_physical * num_y_physical);
    CUFFT_EXEC_C2R(c2r_plan, fourier_coeff_structure_function_part_2->get_data() , physical_part_2);
	abs_and_square<<< grid_dim, block_dim >>>(physical_part_2, num_x_physical * num_y_physical, ((CUDA_FLOAT_REAL) order)/2.0);


	//transform to fourier space: | T(x+r_x,y+r_y, 0)-T(x,y, 0) |^{ \frac{order}{2} }
	matrix_device* part_1_fourier = new matrix_device(dim); 
	matrix_device* part_2_fourier = new matrix_device(dim); 
    CUFFT_EXEC_R2C(r2c_plan, physical_part_1 , part_1_fourier->get_data());
    CUFFT_EXEC_R2C(r2c_plan, physical_part_2 , part_2_fourier->get_data());
	//...rescale the fourier coefficients to get the right values
	scale_on_device<<< part_1_fourier->create_grid_dimension() , part_1_fourier->create_block_dimension() >>>(part_1_fourier->get_data(),1.0/((CUDA_FLOAT_REAL) num_x_physical*num_y_physical) , part_1_fourier->get_data(), columns*rows);
	scale_on_device<<< part_2_fourier->create_grid_dimension() , part_2_fourier->create_block_dimension() >>>(part_2_fourier->get_data(),1.0/((CUDA_FLOAT_REAL) num_x_physical*num_y_physical) , part_2_fourier->get_data(), columns*rows);
	cufftDestroy(c2r_plan);
	cufftDestroy(r2c_plan);

	//build scalar product of them
	int fourier_col = part_1_fourier->get_matrix_dimension().at(0);
	int fourier_row = part_1_fourier->get_matrix_dimension().at(1);
	mult_conj_pointwise_on_device<<<part_1_fourier->create_grid_dimension() , part_1_fourier->create_block_dimension()>>>(part_1_fourier->get_data(),part_2_fourier->get_data() , part_1_fourier->get_data(), fourier_col*fourier_row);
	//...now reduce part_1_fourier
	reduce(part_1_fourier->get_data(), fourier_col * fourier_row, output );

	//free memory
	delete fourier_coeff_structure_function;
	delete fourier_coeff_structure_function_part_2;
	cudaFree(physical_part_1);
	cudaFree(physical_part_2);
	delete part_1_fourier;
	delete part_2_fourier;
}

/**
* builds the sum of all entries in input_used_as_memory and writes the result to output[0]
* @param number_of_entries specifies the number of entries in input_used_as_memory and it has to be a power of 2!
* @param output is the result field, which has to be of size 1
* ATTENTION: input_used_as_memory is the input array, but it will be used as a memory field for computations and therefore its entries are modified!
*/
void structure_function_2d::reduce(CUDA_FLOAT* input_used_as_memory, int number_of_entries, CUDA_FLOAT* output){
	
	//iterate to reduce
	for(int size = number_of_entries; size > 1; ){

		//start reduction with current size
		dim3 block_dim = create_block_dimension_real_matrix_structure_function(size);
		dim3 grid_dim = create_grid_dimension_real_matrix_structure_function(size);
		reduce_step<<< grid_dim , block_dim >>>(input_used_as_memory, size, input_used_as_memory);

		//get the new size of the vector, and make sure, that there will be no missed element
		if(size % 2 == 0)
			size=size/2;
		else
			size=size/2+1;
	}


	//now copy the result to ouput 
	dim3 block_dim = create_block_dimension_real_matrix_structure_function(1);
	dim3 grid_dim = create_grid_dimension_real_matrix_structure_function(1);
	reduce_step<<< grid_dim , block_dim >>>(input_used_as_memory, 1, output);

}



__global__ void reduce_step(CUDA_FLOAT* input, int size, CUDA_FLOAT* output){
	int total_index = get_global_index_struct_function();

	//...the length of the resulting vector
	int current_reduced_size = size/2;
	if(size % 2 == 1)
		current_reduced_size = size/2+1;

	//now check if we are inside the length range of the vector and build shifted indices
	if(total_index < current_reduced_size) {
		int current_index = total_index;
		int shifted_index = total_index + current_reduced_size;

		output[current_index].x = input[current_index].x; 
		output[current_index].y = input[current_index].y; 

		if(shifted_index < size){
			output[current_index].x += input[shifted_index].x; 
			output[current_index].y += input[shifted_index].y; 
		}
	}

}


/**
*
*/
__global__ void abs_and_square(CUDA_FLOAT_REAL* input_and_output, int number_of_real_matrix_elements, CUDA_FLOAT_REAL exponent){
	int total_index = get_global_index_struct_function();

	if(total_index < number_of_real_matrix_elements) {
		input_and_output[total_index] = pow(input_and_output[total_index], exponent);
	}
}

/**
* computes the twiddled input field for the given displacement!
* ATTENTION: input is array of size columns x rows x matrices while ouput is of size columns x rows x 1 !!!!
* ATTENTION: input != ouput is neccessary
*/
__global__ void twiddle_by_shift_temperature(CUDA_FLOAT* input, int columns, int rows, int matrices, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, CUDA_FLOAT* output, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z) {
	int total_index = get_global_index_struct_function();
	int col = get_current_matrix_index_struct_function(total_index, 0, columns, rows, matrices);
	int ro = get_current_matrix_index_struct_function(total_index, 1, columns, rows, matrices);
	int mat = get_current_matrix_index_struct_function(total_index, 2, columns, rows, matrices);

	if(total_index < columns*rows*matrices && mat == 0) {

		//init fields
		output[total_index].x = 0.0;
		output[total_index].y = 0.0;

		//loop over vertical ansatz functions
		for(int m = 0; m < matrices; m++){
			//create wavenumbers for the current matrix entry
			CUDA_FLOAT_REAL wave_number_x = get_wave_number_by_index_real_matrix_structure_function(0, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
			CUDA_FLOAT_REAL wave_number_y = get_wave_number_by_index_real_matrix_structure_function(1, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
		
			//the twiddle factor
			CUDA_FLOAT_REAL arg = r_1 * wave_number_x + r_2*wave_number_y;
			CUDA_FLOAT_REAL abs = 1;
			CUDA_FLOAT_REAL real_twiddle = cos(arg)/abs;
			CUDA_FLOAT_REAL imag_twiddle = sin(arg)/abs;	
			real_twiddle = real_twiddle - 1;	
	
			//now compute the new ouput
			int it_index = col + ro * columns + m*columns*rows;
			CUDA_FLOAT_REAL twiddled_real = ( input[it_index].x * real_twiddle - input[it_index].y * imag_twiddle );
			CUDA_FLOAT_REAL twiddled_imag = ( input[it_index].y * real_twiddle + input[it_index].x * imag_twiddle );
			//second vertical ansatz function is zero at z=0
			if(m == 0){
				output[total_index].x += twiddled_real;
				output[total_index].y += twiddled_imag;
			}
		}
	}

}


/**
* computes the twiddled input field for the given displacement!
* ATTENTION: input is array of size columns x rows x matrices while ouput is of size columns x rows x 1 !!!!
* ATTENTION: input != ouput is neccessary
*/
__global__ void twiddle_by_shift_u_1(CUDA_FLOAT* f_input, CUDA_FLOAT* g_input, int columns, int rows, int matrices, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, CUDA_FLOAT* output, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z) {
	int total_index = get_global_index_struct_function();
	int col = get_current_matrix_index_struct_function(total_index, 0, columns, rows, matrices);
	int ro = get_current_matrix_index_struct_function(total_index, 1, columns, rows, matrices);
	int mat = get_current_matrix_index_struct_function(total_index, 2, columns, rows, matrices);

	if(total_index < columns*rows*matrices && mat == 0) {

		//init fields
		output[total_index].x = 0.0;
		output[total_index].y = 0.0;

		//part of f
		//...loop over vertical ansatz functions
		for(int m = 0; m < matrices; m++){
			//create wavenumbers for the current matrix entry
			CUDA_FLOAT_REAL wave_number_x = get_wave_number_by_index_real_matrix_structure_function(0, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
			CUDA_FLOAT_REAL wave_number_y = get_wave_number_by_index_real_matrix_structure_function(1, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
		
			//the twiddle factor
			CUDA_FLOAT_REAL arg = r_1 * wave_number_x + r_2*wave_number_y;
			CUDA_FLOAT_REAL abs = 1;
			CUDA_FLOAT_REAL real_twiddle = cos(arg)/abs;
			CUDA_FLOAT_REAL imag_twiddle = sin(arg)/abs;	
			real_twiddle = real_twiddle - 1;		

			//calculate the z=0 value of m-th ansatz function \partial_z C_m(z)
			CUDA_FLOAT_REAL factor = 0.0;
			CUDA_FLOAT_REAL z_value = 0.0;
			//TODO
			if(m == 0) {
				factor = 4.73004074;
				z_value = factor * sinh(factor * 0) / cosh(factor * 0.5) + factor * sin(factor * 0) / cos(factor * 0.5);
			}else if(m==1){
				factor = 7.85320462;	
				z_value = factor * cosh(factor * 0) / sinh(factor * 0.5) -  factor * cos(factor * 0) / sin(factor * 0.5);
			}
	
			//build twiddled part and multiply by z value
			int it_index = col + ro * columns + m*columns*rows;
			CUDA_FLOAT to_add;
			to_add.x = ( f_input[it_index].x * real_twiddle - f_input[it_index].y * imag_twiddle ) * z_value;
			to_add.y = ( f_input[it_index].y * real_twiddle + f_input[it_index].x * imag_twiddle ) * z_value;
			to_add.x *= wave_number_x;
			to_add.y *= wave_number_x;

			//build x derivative for f
			output[total_index].x += (-1.0) * to_add.y;
			output[total_index].y += to_add.x;
			
		}

		//part of g
		//...loop over vertical ansatz functions
		for(int m = 0; m < matrices; m++){
			//create wavenumbers for the current matrix entry
			CUDA_FLOAT_REAL wave_number_x = get_wave_number_by_index_real_matrix_structure_function(0, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
			CUDA_FLOAT_REAL wave_number_y = get_wave_number_by_index_real_matrix_structure_function(1, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
		
			//the twiddle factor
			CUDA_FLOAT_REAL arg = r_1 * wave_number_x + r_2*wave_number_y;
			CUDA_FLOAT_REAL abs = 1;
			CUDA_FLOAT_REAL real_twiddle = cos(arg)/abs;
			CUDA_FLOAT_REAL imag_twiddle = sin(arg)/abs;	
			real_twiddle = real_twiddle - 1;		
	
			//build twiddled part and multiply by z value
			int it_index = col + ro * columns + m*columns*rows;
			CUDA_FLOAT to_add;
			to_add.x = ( g_input[it_index].x * real_twiddle - g_input[it_index].y * imag_twiddle );
			to_add.y = ( g_input[it_index].y * real_twiddle + g_input[it_index].x * imag_twiddle );

			//build x derivative for f
			//second vertical ansatz function is zero at z=0
			if(m == 0){
				output[total_index].x += (-1.0) * to_add.y * wave_number_y;
				output[total_index].y += to_add.x * wave_number_y;
			}
		}

	}

}


/**
* computes the twiddled input field for the given displacement!
* ATTENTION: input is array of size columns x rows x matrices while ouput is of size columns x rows x 1 !!!!
* ATTENTION: input != ouput is neccessary
*/
__global__ void twiddle_by_shift_u_2(CUDA_FLOAT* f_input, CUDA_FLOAT* g_input, int columns, int rows, int matrices, CUDA_FLOAT_REAL r_1, CUDA_FLOAT_REAL r_2, CUDA_FLOAT* output, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z) {
	int total_index = get_global_index_struct_function();
	int col = get_current_matrix_index_struct_function(total_index, 0, columns, rows, matrices);
	int ro = get_current_matrix_index_struct_function(total_index, 1, columns, rows, matrices);
	int mat = get_current_matrix_index_struct_function(total_index, 2, columns, rows, matrices);

	if(total_index < columns*rows*matrices && mat == 0) {

		//init fields
		output[total_index].x = 0.0;
		output[total_index].y = 0.0;

		//part of f
		//...loop over vertical ansatz functions
		for(int m = 0; m < matrices; m++){
			//create wavenumbers for the current matrix entry
			CUDA_FLOAT_REAL wave_number_x = get_wave_number_by_index_real_matrix_structure_function(0, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
			CUDA_FLOAT_REAL wave_number_y = get_wave_number_by_index_real_matrix_structure_function(1, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
		
			//the twiddle factor
			CUDA_FLOAT_REAL arg = r_1 * wave_number_x + r_2*wave_number_y;
			CUDA_FLOAT_REAL abs = 1;
			CUDA_FLOAT_REAL real_twiddle = cos(arg)/abs;
			CUDA_FLOAT_REAL imag_twiddle = sin(arg)/abs;	
			real_twiddle = real_twiddle - 1;		

			//calculate the z=0 value of m-th ansatz function \partial_z C_m(z)
			CUDA_FLOAT_REAL factor = 0.0;
			CUDA_FLOAT_REAL z_value = 0.0;
			//TODO
			if(m == 0) {
				factor = 4.73004074;
				z_value = factor * sinh(factor * 0) / cosh(factor * 0.5) + factor * sin(factor * 0) / cos(factor * 0.5);
			}else if(m==1){
				factor = 7.85320462;	
				z_value = factor * cosh(factor * 0) / sinh(factor * 0.5) -  factor * cos(factor * 0) / sin(factor * 0.5);
			}
	
			//build twiddled part and multiply by z value
			int it_index = col + ro * columns + m*columns*rows;
			CUDA_FLOAT to_add;
			to_add.x = ( f_input[it_index].x * real_twiddle - f_input[it_index].y * imag_twiddle ) * z_value;
			to_add.y = ( f_input[it_index].y * real_twiddle + f_input[it_index].x * imag_twiddle ) * z_value;
			to_add.x *= wave_number_y;
			to_add.y *= wave_number_y;

			//build x derivative for f
			output[total_index].x += (-1.0) * to_add.y;
			output[total_index].y += to_add.x;
			
		}

		//part of g
		//...loop over vertical ansatz functions
		for(int m = 0; m < matrices; m++){
			//create wavenumbers for the current matrix entry
			CUDA_FLOAT_REAL wave_number_x = get_wave_number_by_index_real_matrix_structure_function(0, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
			CUDA_FLOAT_REAL wave_number_y = get_wave_number_by_index_real_matrix_structure_function(1, col, ro, m, columns, rows, matrices, cube_length_x, cube_length_y, cube_length_z);
		
			//the twiddle factor
			CUDA_FLOAT_REAL arg = r_1 * wave_number_x + r_2*wave_number_y;
			CUDA_FLOAT_REAL abs = 1;
			CUDA_FLOAT_REAL real_twiddle = cos(arg)/abs;
			CUDA_FLOAT_REAL imag_twiddle = sin(arg)/abs;	
			real_twiddle = real_twiddle - 1;		
	
			//build twiddled part and multiply by z value
			int it_index = col + ro * columns + m*columns*rows;
			CUDA_FLOAT to_add;
			to_add.x = ( g_input[it_index].x * real_twiddle - g_input[it_index].y * imag_twiddle );
			to_add.y = ( g_input[it_index].y * real_twiddle + g_input[it_index].x * imag_twiddle );

			//build x derivative for f
			//second vertical ansatz function is zero at z=0
			if(m == 0){
				output[total_index].x -= (-1.0) * to_add.y * wave_number_x;
				output[total_index].y -= to_add.x * wave_number_x;
			}
		}

	}

}


__device__ int get_global_index_struct_function(){
	return threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y + blockIdx.x*blockDim.x*blockDim.y*blockDim.z + blockIdx.y*blockDim.x*blockDim.y*blockDim.z*gridDim.x + blockIdx.z*blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y;	
}

__device__ int get_current_matrix_index_struct_function(int total_index, int logic_index, int columns, int rows, int matrices){
	int mod = columns;
	int div = 1;
	if(logic_index == 0)
		return ((total_index % mod) / div);
	mod *= rows;
	div *= columns;
	if(logic_index == 1)
		return ((total_index % mod) / div);
	mod *= matrices;
	div *= rows;
	if(logic_index == 2)
		return ((total_index % mod) / div);

	return 0;	
}


__device__ CUDA_FLOAT_REAL get_wave_number_by_index_real_matrix_structure_function(int logic_index, int col, int ro, int mat, int columns, int rows, int matrices, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z){
	//...columns	
	int wave_number_col;
	wave_number_col = col; 	
	//...rows
	int wave_number_row;
	if(ro < (rows/2)+1) 
		wave_number_row = ro; 
	else
		wave_number_row = (-1)*(rows - ro);
	//...matrices
	int wave_number_matrix;
	if(mat < (matrices/2)+1) 
		wave_number_matrix = mat; 
	else
		wave_number_matrix = (-1)*(matrices - mat);

	CUDA_FLOAT_REAL return_value;
	if(logic_index == 0)
		return_value = wave_number_col*2*M_PI / cube_length_x;
	else if (logic_index == 1)
		return_value = wave_number_row*2*M_PI / cube_length_y;
	else
		return_value = wave_number_matrix*2*M_PI / cube_length_z;
	
	return return_value;
}


__host__ dim3 create_block_dimension_real_matrix_structure_function(int number_of_matrix_entries){
	dim3 block;
	block.x = MAX_NUMBER_THREADS_PER_BLOCK;
	return block;
}


__host__ dim3 create_grid_dimension_real_matrix_structure_function(int number_of_matrix_entries){
	dim3 grid;
	
	int n = number_of_matrix_entries;
	if(n % MAX_NUMBER_THREADS_PER_BLOCK == 0)
		grid.x = n/MAX_NUMBER_THREADS_PER_BLOCK;
	else
		grid.x = (n/MAX_NUMBER_THREADS_PER_BLOCK)+1;

	return grid;
}

__global__ void mult_conj_pointwise_on_device(CUDA_FLOAT* input,CUDA_FLOAT* to_mult , CUDA_FLOAT* result, int number_of_matrix_entries){
	int index = get_global_index_struct_function();
	if(index < number_of_matrix_entries) {
		CUDA_FLOAT a = input[index];
		CUDA_FLOAT b = to_mult[index];

		//...complex conjugate of b
		b.y = (-1.0) * b.y;

		result[index].x =  a.x * b.x - a.y * b.y;
		result[index].y =  a.x * b.y + a.y * b.x;
	}
}


/*!
* performs result = factor * input
* input == result is allowed
*/
__global__ void scale_on_device(CUDA_FLOAT* input,CUDA_FLOAT_REAL factor , CUDA_FLOAT* result, int number_of_matrix_entries){
    int index = get_global_index_struct_function();
    if(index < number_of_matrix_entries) {
        CUDA_FLOAT a = input[index];

        result[index].x =  a.x * factor;
        result[index].y =  a.y * factor;
    }
}


