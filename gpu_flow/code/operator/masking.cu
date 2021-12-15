#include "masking.h"

masking::masking(std::vector<int> dimension) {

    const int num_x = 2*(dimension.at(0)-1);
    const int num_y = dimension.at(1);
    c_x = 0.5;
    c_y = 0.5;
    radius = 0.15;
    width = 0.05;
    phector = 1.1;
    // Allocate mask memory
    if(cudaSuccess != cudaMalloc((void**) &ze_mask, sizeof(int) * num_x * num_y)) {
        EXIT_ERROR("not able to allocate mask memory");
    }

    //create fft plan
    cufftPlan2d(&c2r_plan, num_y, num_x, CUFFT_TYPE_C2R);
    //create inverse fft plan
    cufftPlan2d(&r2c_plan, num_y, num_x, CUFFT_TYPE_R2C);
}

masking* masking::init_the_mask(std::vector<int> dimension){
	masking* hehe = new masking(dimension);
	return hehe;
}

void masking::set_mask_parameters(CUDA_FLOAT_REAL x, CUDA_FLOAT_REAL y, CUDA_FLOAT_REAL r, CUDA_FLOAT_REAL factor) {
	c_x = x;
	c_y = y;
	radius = r;
	phector = factor;
}

masking::~masking(){
	cudaFree(ze_mask);
	delete theta_data;
	//free cuda fft plans
	cufftDestroy(c2r_plan);
	cufftDestroy(r2c_plan);
}
/*
__global__ static void apply_mask(CUDA_FLOAT_REAL* input_output, CUDA_FLOAT_REAL* mask_data, int* mask, int num_xy, int num_entries)
{
    int total_index = get_global_index();

    // check if thread is valid
    if(total_index < num_entries) {

        CUDA_FLOAT_REAL entry;
        if(mask[total_index % num_xy] == 0) {
            entry = input_output[total_index];
        } else {
		entry = mask_data[total_index];
        }
		input_output[total_index] = entry;
	}
}
*/


//applies the mask
__global__ static void julie_do_the_thing(CUDA_FLOAT_REAL* mask_data, int* mask, int num_xy, int num_entries, CUDA_FLOAT_REAL factor, CUDA_FLOAT_REAL phector){
	int total_index = get_global_index();
	// check if thread is valid
	if(total_index < num_entries) {
		if(mask[total_index % num_xy] == 0) {
//last changes
			mask_data[total_index] = factor*mask_data[total_index]*phector; //add temperature
		} else {
			mask_data[total_index] *= factor;
		}
	}
}

//q : are both n e c e s s a r y ?

void masking::masker(matrix_folder* theta) {
	// Get number of modes

	std::vector<int> theta_dim = theta->get_matrix(0)->get_matrix_dimension();

	const int mox = theta_dim.at(0);	// modes in x direction
	const int moy = theta_dim.at(1);	// modes in y direction
	const int moz = theta_dim.at(2);	// modes in z direction
	const int moxy = mox * moy;	// modes in horizontal plane
	const int num_elements_real = 2 * (mox - 1) * moy;	// number of elements in real space
	

	//bool quit = false;

	// prepare number of real modes
	std::vector<int> dim_real(3);
	dim_real[0] = 2 * (mox - 1);
	dim_real[1] = moy;
	dim_real[2] = moz;

	theta_data = new matrix_device_real(dim_real);


		// Transform theta

		for (int i = 0; i < moz; i++) {
			if (CUFFT_SUCCESS != CUFFT_EXEC_C2R(c2r_plan, theta->get_matrix(0)->get_data() + i * moxy, theta_data->get_data() + i * num_elements_real)) {
				DBGSYNC();
				EXIT_ERROR2("c2r-fft failed", ::cudaGetErrorString(cudaGetLastError()));
			}
		}
		const dim3 grid_ze_mask = create_grid_dim(num_elements_real);
		const dim3 block_ze_mask = create_block_dim(num_elements_real);
		const dim3 grid_heat = create_grid_dim(num_elements_real * moz);
    		const dim3 block_heat = create_block_dim(num_elements_real * moz);

		//create mask_data
		init_mask_physical_space_circle<<<grid_ze_mask,block_ze_mask>>>(ze_mask, dim_real[0], dim_real[1], radius, c_x, c_y);
		
		//init_mask_physical_space_rectangle<<<grid_ze_mask,block_ze_mask>>>(ze_mask, dim_real[0], dim_real[1], width); 
		//needs more work to see where to change exactly
		
		//the mask is recalculated at ea iteration : that is not necessary maybe

		//apply_mask
		julie_do_the_thing<<<grid_heat,block_heat>>>(theta_data->get_data(), ze_mask, num_elements_real, num_elements_real * moz, 1./num_elements_real, phector);			


		// Transform theta back
		for (int i = 0; i < moz; i++) {
			if (CUFFT_SUCCESS != CUFFT_EXEC_R2C(r2c_plan, theta_data->get_data() + i * num_elements_real, theta->get_matrix(0)->get_data() + i * moxy)) {
				DBGSYNC();
				EXIT_ERROR2("r2c-fft failed", ::cudaGetErrorString(cudaGetLastError()));
			}
		}
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

// radius in fraction of max radius
__global__ static void init_mask_physical_space_circle(int* mask, int columns, int rows, CUDA_FLOAT_REAL radius, CUDA_FLOAT_REAL c_x, CUDA_FLOAT_REAL c_y){
    int total_index = get_global_index();
    int col_index = 0, row_index = 0, matrix_index = 0;
    get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, 1);

	//use a mask
	if(total_index < columns*rows){
        // init all with zeros
        int mask_val = 0;

        // circular mask
        CUDA_FLOAT_REAL distance_from_center_sq = (col_index - c_y*columns)*(col_index - c_y*columns) + (row_index - c_x*rows)*(row_index - c_x*rows);

        int min_len = (rows < columns)?(rows):(columns);
        CUDA_FLOAT_REAL max_distance_from_center = radius * 0.5 * min_len;
        if(distance_from_center_sq > max_distance_from_center * max_distance_from_center) {
            mask_val = 1;
        }

        mask[total_index] = mask_val;
	}
}

// width in fraction of edge length
__global__ static void init_mask_physical_space_rectangle(int* mask, int columns, int rows, CUDA_FLOAT_REAL width){
    int total_index = get_global_index();
    int col_index = 0, row_index = 0, matrix_index = 0;
    get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, 1);

    // use a mask
	if(total_index < columns*rows){
		//init all with zeros
        int mask_val = 0;

        // rectangular mask
        if((col_index < width * columns)
                || (col_index > (1-width) * columns)
                || (row_index < width * rows)
                || (row_index > (1-width) * rows)) {
            mask_val = 1;
        }

        mask[total_index] = mask_val;
	}
}

