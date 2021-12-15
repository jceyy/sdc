#include "masktemp.h"


/*****************************************************************************/
/*************************Heat source at line 201*****************************/
/*****************************************************************************/
masktemp::masktemp(std::vector<int> dimension) : tempadded(false){

    const int num_x = 2*(dimension.at(0)-1);
    const int num_y = dimension.at(1);

    // Allocate mask memory
    if(cudaSuccess != cudaMalloc((void**) &mask_tempadded, sizeof(int) * num_x * num_y)) {
        EXIT_ERROR("not able to allocate mask memory");
    }

    //create fft plan
    cufftPlan2d(&c2r_plan, num_y, num_x, CUFFT_TYPE_C2R);
    //create inverse fft plan
    cufftPlan2d(&r2c_plan, num_y, num_x, CUFFT_TYPE_R2C);
}

masktemp::~masktemp(){
    cudaFree(mask_tempadded);

    if(tempadded) {
        delete theta_tempadded;
    
    }

    //free cuda fft plans
    cufftDestroy(c2r_plan);
    cufftDestroy(r2c_plan);
}

void masktemp::temp_circle(matrix_folder* theta, CUDA_FLOAT_REAL radius) {

    // Get number of modes
    std::vector<int> f_dim = theta->get_matrix(0)->get_matrix_dimension();
    const int num_x = 2 * (f_dim.at(0)-1),
              num_y = f_dim.at(1),
              num_xy = num_x * num_y;

    init_mask_physical_space_circle<<<create_grid_dim(num_xy), create_block_dim(num_xy)>>> (mask_tempadded, num_x, num_y, radius);

    temp_mask(theta);
}

void masktemp::temp_rectangle(matrix_folder* theta, CUDA_FLOAT_REAL width) {

    // Get number of modes
    std::vector<int> f_dim = theta->get_matrix(0)->get_matrix_dimension();
    const int num_x = 2 * (f_dim.at(0)-1),
              num_y = f_dim.at(1),
              num_xy = num_x * num_y;

    init_mask_physical_space_rectangle<<<create_grid_dim(num_xy), create_block_dim(num_xy)>>> (mask_tempadded, num_x, num_y, width);

    temp_mask(theta);
}

void masktemp::temp_mask(matrix_folder* theta) {

    // Clear old field
    if(tempadded) {
        delete theta_tempadded;
     
    }

    // Get number of modes
    std::vector<int> f_dim = theta->get_matrix(0)->get_matrix_dimension();
    const int mox = f_dim.at(0),
              moy = f_dim.at(1),
              moz = f_dim.at(2),
              moxy = mox * moy,
              num_elements_real = 2 * (mox-1) * moy;

    // prepare number of real modes
    std::vector<int> dim_real(3);
    dim_real[0] = 2 * (mox-1);
    dim_real[1] = moy;
    dim_real[2] = moz;
    theta_tempadded = new matrix_device_real(dim_real);


    // Transform theta
    for(int i = 0; i < moz; i++) {
        if(CUFFT_SUCCESS != CUFFT_EXEC_C2R(c2r_plan, theta->get_matrix(0)->get_data() + i*moxy, theta_tempadded->get_data() + i*num_elements_real) ) {
            DBGSYNC();
            EXIT_ERROR2("c2r-fft failed", ::cudaGetErrorString(cudaGetLastError()));
        }
    }

    tempadded = true;
}


masktemp* masktemp::init_operator(std::vector<int> dimension){
    masktemp* op = new masktemp(dimension);
    return op;
}


void masktemp::calculate_operator(matrix_folder* theta){

    if(!tempadded) return;

    // build u_i^{(1)}(x,y,n,t) for i = 1,2,3
    std::vector<int> f_dim = theta->get_matrix(0)->get_matrix_dimension();
    const int mox = f_dim.at(0),    // modes in x direction
              moy = f_dim.at(1),    // modes in y direction
              moz = f_dim.at(2);    // modes in z direction
    const int moxy = mox * moy;     // modes in horizontal plane
    const int num_elements_real = 2 * (mox-1) * moy;    // number of elements in real space


    // prepare number of real modes
    std::vector<int> dim_real(3);
    dim_real[0] = 2 * (mox-1);
    dim_real[1] = moy;
    dim_real[2] = moz;
    matrix_device_real* theta_real = new matrix_device_real(dim_real);

    // Transform theta
    for(int i = 0; i < moz; i++) {
        if(CUFFT_SUCCESS != CUFFT_EXEC_C2R(c2r_plan, theta->get_matrix(0)->get_data() + i*moxy, theta_real->get_data() + i*num_elements_real) ) {
            DBGSYNC();
            EXIT_ERROR2("c2r-fft failed", ::cudaGetErrorString(cudaGetLastError()));
        }
    }




    // Apply mask
    const dim3 grid_real = create_grid_dim(num_elements_real * moz);
    const dim3 block_real = create_block_dim(num_elements_real * moz);
//m change
    apply_mask<<<grid_real,block_real>>>(theta_real->get_data(), theta_tempadded->get_data(), mask_tempadded, 1./num_elements_real, num_elements_real, num_elements_real * moz);

    /*theta_real->scale_itself(1./num_elements_real);
    f_real->scale_itself(1./num_elements_real);
    g_real->scale_itself(1./num_elements_real);*/

    // Transform theta back
    for(int i = 0; i < moz; i++) {
        if(CUFFT_SUCCESS != CUFFT_EXEC_R2C(r2c_plan, theta_real->get_data() + i*num_elements_real, theta->get_matrix(0)->get_data() + i*moxy) ) {
            DBGSYNC();
            EXIT_ERROR2("r2c-fft failed", ::cudaGetErrorString(cudaGetLastError()));
        }
    }


    delete theta_real;
   

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



__global__ static void apply_mask(CUDA_FLOAT_REAL* input_output, CUDA_FLOAT_REAL* mask_data, int* mask, CUDA_FLOAT_REAL factor, int num_xy, int num_entries){
    int total_index = get_global_index();

// check if thread is valid
    if(total_index < num_entries) {

        CUDA_FLOAT_REAL entry;
        if(mask[total_index % num_xy] == 1) {
            entry = input_output[total_index];
        } else {
            entry = (input_output[total_index]+1.5/100); //This is where the heat source will add temperature
        }
        input_output[total_index] = factor * entry;
	}
}

//q : are both n e c e s s a r y ?

// width in fraction of edge length
//not necessary for the mom
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

// radius in fraction of max radius
__global__ static void init_mask_physical_space_circle(int* mask, int columns, int rows, CUDA_FLOAT_REAL radius){
    int total_index = get_global_index();
    int col_index = 0, row_index = 0, matrix_index = 0;
    get_current_matrix_indices(col_index, row_index, matrix_index, total_index, columns, rows, 1);

	//use a mask
	if(total_index < columns*rows){
        // init all with zeros
        int mask_val = 0;

        // circular mask
        //CUDA_FLOAT_REAL distance_from_center_sq = (col_index - 0.3 *  columns)*(col_index - 0.3 * columns) + (row_index - 0.3 * columns)*(row_index - 0.3 * rows);	//maybe change
	//m change
        CUDA_FLOAT_REAL distance_from_center_sq = (col_index - 0.2 *  columns)*(col_index - 0.2 * columns) + (row_index - 0.3 * rows)*(row_index - 0.3 * rows);	//maybe change
        int min_len = (rows < columns)?(rows):(columns);
        CUDA_FLOAT_REAL max_distance_from_center = radius * 0.5 * min_len;
        if(distance_from_center_sq > max_distance_from_center * max_distance_from_center) {
            mask_val = 1;
        }

        mask[total_index] = mask_val;
	}
}



