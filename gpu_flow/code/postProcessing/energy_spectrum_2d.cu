#include "energy_spectrum_2d.h"

energy_spectrum_2d::energy_spectrum_2d(std::vector<int> dimensions, std::vector<CUDA_FLOAT_REAL> cube_length){
	this->dim_x = dimensions.at(0);
	this->dim_y = dimensions.at(1);
	this->dim_z = dimensions.at(2);

	this->cube_length_x = cube_length.at(0);
	this->cube_length_y = cube_length.at(1);
	this->cube_length_z = cube_length.at(2);
	
}


energy_spectrum_2d::~energy_spectrum_2d(){


}

matrix_folder* energy_spectrum_2d::calculate_energy_spectrum(matrix_folder* velocity_at_z_zero){
	int folder_dim = velocity_at_z_zero->get_dimension();

	matrix_folder* return_folder = new matrix_folder(folder_dim);

	for(int i = 0; i < folder_dim; i++) {
		int num_x_real = velocity_at_z_zero->get_matrix(i)->get_matrix_dimension().at(0);
		int num_y_real = velocity_at_z_zero->get_matrix(i)->get_matrix_dimension().at(1);

		//create fft plans
		cufftHandle c2r_plan;
		cufftHandle r2c_plan;
		cufftPlan2d(&c2r_plan,num_y_real,num_x_real,CUFFT_C2R);
		cufftPlan2d(&r2c_plan,num_y_real,num_x_real,CUFFT_R2C);

		//...create real data field to perform r2c transformation
		CUDA_FLOAT_REAL* real_data;
		if(cudaSuccess != cudaMalloc((void**) &real_data, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real)) {
            DBGOUT("not able to allocate temporary real data to calculate energy spectrum!");
		}
		dim3 grid_dim = create_grid_dimension_real_matrix_energy_spectrum(num_x_real * num_y_real);
		dim3 block_dim = create_block_dimension_real_matrix_energy_spectrum(num_x_real * num_y_real);
		copy_complex_data_to_real_data<<<grid_dim,block_dim>>>(velocity_at_z_zero->get_matrix(i)->get_data(), num_x_real*num_y_real , real_data );

		//...create output field for Fourier transformation
		CUDA_FLOAT* fft_output;
		int num_x_fourier = num_x_real/2 + 1;
		int num_y_fourier = num_y_real;
		if(cudaSuccess != cudaMalloc((void**) &fft_output, sizeof(CUDA_FLOAT) * num_x_fourier * num_y_fourier)) {
            DBGOUT("not able to allocate memory for FFT output to calculate energy spectrum!");
		}

        CUFFT_EXEC_R2C(r2c_plan, real_data , fft_output);
		dim3 grid_dim_fourier = create_grid_dimension_real_matrix_energy_spectrum(num_x_fourier * num_y_fourier);
		dim3 block_dim_fourier = create_block_dimension_real_matrix_energy_spectrum(num_x_fourier * num_y_fourier);

		//TODO scale results to get real amplitudes
		scale_on_device_complex_spectrum<<<grid_dim, block_dim>>>(fft_output,1.0/(num_x_real*num_y_real) , fft_output, num_x_fourier * num_y_fourier);

		//implement energy spectrum for each matrix
		//...build the spectrum		
		int energy_spectrum_size = num_x_fourier;
		if(energy_spectrum_size < num_y_fourier/2+1)
			energy_spectrum_size = num_y_fourier/2+1;
		std::vector<int> spectrum_dimension; 
		spectrum_dimension.push_back(energy_spectrum_size);spectrum_dimension.push_back(1);spectrum_dimension.push_back(1);
		matrix_device* current_spectrum = new matrix_device(spectrum_dimension);
		//create spectrum with 2-norm
		reduce_spectrum<<<grid_dim_fourier,block_dim_fourier>>>(fft_output,num_x_fourier,num_y_fourier, this->cube_length_x, this->cube_length_y, this->cube_length_z,current_spectrum->get_data(), energy_spectrum_size );

		//set return value
		return_folder->add_matrix(i, current_spectrum);

		//free temporary memory
		cudaFree(real_data);
		cudaFree(fft_output);
		//free cuda fft plans
		cufftDestroy(c2r_plan);
		cufftDestroy(r2c_plan);
	}

	//set return value
	return return_folder;
}

__global__ void reduce_spectrum(CUDA_FLOAT* fft_output,int num_x_fourier, int num_y_fourier, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y, CUDA_FLOAT_REAL cube_length_z, CUDA_FLOAT* spectrum_output, int energy_spectrum_size ){
	int total_index = get_global_index_energy_spectrum();
	
	//max wave index
	int max_x_index = num_x_fourier-1;
	int max_y_index = num_y_fourier/2;

	//mode with highest magnitude
	CUDA_FLOAT_REAL max_mode_number_x = 2.0 * M_PI * (max_x_index) / cube_length_x; 
	CUDA_FLOAT_REAL max_mode_number_y = 2.0 * M_PI * (max_y_index) / cube_length_y;
	CUDA_FLOAT_REAL max_mode_abs = sqrt( (max_mode_number_x*max_mode_number_x) + (max_mode_number_y*max_mode_number_y) );
	CUDA_FLOAT_REAL mode_increment = max_mode_abs / ((CUDA_FLOAT_REAL) energy_spectrum_size);

	if(total_index < energy_spectrum_size ){

		spectrum_output[total_index].x = 0;		
		spectrum_output[total_index].y = 0;

		CUDA_FLOAT_REAL this_mode_abs = total_index * mode_increment;
		CUDA_FLOAT_REAL next_mode_abs = (total_index+1) * mode_increment;

		for(int i = 0; i < num_x_fourier; i++) {
			for(int j = 0; j < num_y_fourier; j++) {
				//the current mode
				CUDA_FLOAT_REAL current_mode_x = 2.0 * M_PI * (i) / cube_length_x; 
				CUDA_FLOAT_REAL current_mode_y;				
				if(j <= num_y_fourier/2)
					current_mode_y = 2.0 * M_PI * (j) / cube_length_y; 
				else
					current_mode_y = 2.0 * M_PI * (num_y_fourier-j) / cube_length_y; 

				//magnitude of the current mode
				CUDA_FLOAT_REAL current_mode_abs = sqrt(current_mode_x*current_mode_x + current_mode_y*current_mode_y);

				//check if this mode is inside the current range
				if(this_mode_abs <= current_mode_abs && next_mode_abs > current_mode_abs){
					int current_index = i + j*num_x_fourier;
					spectrum_output[total_index].x += fft_output[current_index].x*fft_output[current_index].x + fft_output[current_index].y*fft_output[current_index].y;
					spectrum_output[total_index].y = 0.0;
				}
			}
		}
	}

}

__global__ void copy_complex_data_to_real_data(CUDA_FLOAT* input, int number_of_elements , CUDA_FLOAT_REAL* output ){
	int total_index = get_global_index_energy_spectrum();

	if(total_index < number_of_elements) {
		output[total_index] = input[total_index].x;
	}
	
}

__device__ int get_global_index_energy_spectrum(){
	return threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y + blockIdx.x*blockDim.x*blockDim.y*blockDim.z + blockIdx.y*blockDim.x*blockDim.y*blockDim.z*gridDim.x + blockIdx.z*blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y;	
}


__device__ int get_current_matrix_index_energy_spectrum(int total_index, int logic_index, int columns, int rows, int matrices){
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


__host__ dim3 create_block_dimension_real_matrix_energy_spectrum(int number_of_matrix_entries){
	dim3 block;
	block.x = MAX_NUMBER_THREADS_PER_BLOCK;
	return block;
}


__host__ dim3 create_grid_dimension_real_matrix_energy_spectrum(int number_of_matrix_entries){
	dim3 grid;
	
	int n = number_of_matrix_entries;
	if(n % MAX_NUMBER_THREADS_PER_BLOCK == 0)
		grid.x = n/MAX_NUMBER_THREADS_PER_BLOCK;
	else
		grid.x = (n/MAX_NUMBER_THREADS_PER_BLOCK)+1;

	return grid;
}


__global__ void scale_on_device_complex_spectrum(CUDA_FLOAT* input,CUDA_FLOAT_REAL factor , CUDA_FLOAT* result, int number_of_matrix_entries){
	int index = get_global_index_energy_spectrum();
	if(index < number_of_matrix_entries) {
		CUDA_FLOAT a = input[index];
		result[index].x =  a.x * factor ;
		result[index].y =  a.y * factor;
	}
}





