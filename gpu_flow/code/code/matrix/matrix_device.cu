#include "matrix_device.h"

template<class EntryType>
__host__ basic_matrix_device<EntryType>::basic_matrix_device(){
    d_data = NULL;
}

/*!
* creates a new device matrix
* i.e.: allocates memory and copies the data from matrix to the CUDA device
*/
template<class EntryType>
__host__ basic_matrix_device<EntryType>::basic_matrix_device(basic_matrix_host<EntryType>* m, InitEntries initEntries){
    init(m);

    if(initEntries == Copy) {
        //copy data to device
        cudaMemcpy(d_data, m->get_data(), sizeof(EntryType) * m->get_number_of_matrix_entries(), cudaMemcpyHostToDevice);
    }
}

/*!
* creates a new device matrix
* i.e.: allocates memory and copies the data from matrix to the CUDA device
*/
template<class EntryType>
__host__ basic_matrix_device<EntryType>::basic_matrix_device(basic_matrix_device<EntryType>* m, InitEntries initEntries){
    init(m);

    if(initEntries == Copy) {
        //copy data to device
        cudaMemcpy(d_data, m->d_data, sizeof(CUDA_FLOAT) * m->get_number_of_matrix_entries(), cudaMemcpyDeviceToDevice);
    }
}

/*!
* creates a non initialized device matrice with given dimensions
*
*/
template<class EntryType>
__host__ basic_matrix_device<EntryType>::basic_matrix_device(vector<int> dimensions) {
	if(dimensions.empty()) {
        basic_matrix_device<EntryType>();
		return;
	}	

	int num_elements = 1;	
    for(size_t i = 0; i < dimensions.size(); i++) {
		num_elements *= dimensions.at(i);
	}
    if(cudaSuccess != cudaMalloc((void**) &d_data, sizeof(EntryType) * num_elements)) {
        EXIT_ERROR2("not able to allocate device memory for matrix_device", ::cudaGetErrorString(cudaGetLastError()));
	}

	//copy info about dim
    dimension = dimensions;
}

/***********************************************************************************/
/** Memory initializer functions                                                  **/
/***********************************************************************************/
template<class EntryType>
__host__ void basic_matrix_device<EntryType>::init(basic_matrix_host<EntryType> *m) {
    if(cudaSuccess != cudaMalloc((void**) &d_data, sizeof(EntryType) * m->get_number_of_matrix_entries())) {
        EXIT_ERROR2("not able to allocate device memory for matrix_device", ::cudaGetErrorString(cudaGetLastError()));
    }

    //copy information about dimensions
    dimension = m->get_matrix_dimensions();
}

template<class EntryType>
__host__ void basic_matrix_device<EntryType>::init(basic_matrix_device<EntryType>* m) {
    if(cudaSuccess != cudaMalloc((void**) &d_data, sizeof(EntryType) * m->get_number_of_matrix_entries())) {
        EXIT_ERROR2("not able to allocate device memory for matrix_device", ::cudaGetErrorString(cudaGetLastError()));
	}
	
	//copy information about dimensions
    dimension = m->get_matrix_dimension();
}

template<class EntryType>
__host__ basic_matrix_device<EntryType>::~basic_matrix_device() {
    if(cudaSuccess != cudaFree(d_data)) {
        EXIT_ERROR2("not able to free device memory for matrix_device", ::cudaGetErrorString(cudaGetLastError()));
    }
}

template<class EntryType>
__host__ void basic_matrix_device<EntryType>::set_zero() {
    if(d_data == NULL) EXIT_ERROR("device memory not allocated");
    init_zeros<<< create_grid_dimension(), create_block_dimension() >>>(d_data, get_number_of_matrix_entries());
}

template<class EntryType>
__host__ void basic_matrix_device<EntryType>::set_identity() {
    if(d_data == NULL) EXIT_ERROR("device memory not allocated");
    init_ident_matrix<<< create_grid_dimension(), create_block_dimension() >>>(d_data, dimension.at(0), dimension.at(1), get_number_of_matrix_entries());
}

template<class EntryType>
void basic_matrix_device<EntryType>::set_scalar(EntryType val) {
    if(d_data == NULL) EXIT_ERROR("device memory not allocated");
    init_scalar<<< create_grid_dimension(), create_block_dimension() >>>(d_data, val, get_number_of_matrix_entries());
}

template<class EntryType>
__host__ std::vector<int> basic_matrix_device<EntryType>::get_matrix_dimension() const {
    return dimension;
}

template<class EntryType>
__host__ int basic_matrix_device<EntryType>::get_matrix_size(int dimension_index) {
    return dimension.at(dimension_index);
}

template<class EntryType>
int basic_matrix_device<EntryType>::get_number_of_matrix_entries(){
    if(dimension.empty())
		return 0;

	int number_of_entries = 1;
    for(size_t i = 0; i < dimension.size(); i++)
        number_of_entries *= dimension[i];

	return number_of_entries;	
}

/*!
* calculates return = this + (to_add * factor)
* @return returns a new matrix with the result; there is no memory management bis this class done!
*/
template<class EntryType>
__host__ basic_matrix_device<EntryType>* basic_matrix_device<EntryType>::add_mult_pointwise(basic_matrix_device<EntryType>* to_add, CUDA_FLOAT_REAL factor){
    basic_matrix_device<EntryType>* result = new basic_matrix_device<EntryType>(this, basic_matrix_device<EntryType>::noInit);

	//perform matrix arithmetic on cuda device
    add_mult_pointwise_on_device<<<create_grid_dimension(),create_block_dimension()>>>
                                (d_data, to_add->d_data, factor, result->d_data, get_number_of_matrix_entries());

	return result;
}

/*!
* calculates: this = this + (to_add * factor)
*/
template<class EntryType>
__host__ void basic_matrix_device<EntryType>::add_mult_itself_pointwise(basic_matrix_device<EntryType>* to_add, CUDA_FLOAT_REAL factor){
	//perform matrix arithmetic on cuda device
    add_mult_pointwise_on_device<<<create_grid_dimension(),create_block_dimension()>>>(
                                   d_data, to_add->d_data, factor, d_data, get_number_of_matrix_entries());
}

/*!
* calculates this = this * to_mult , where '*' denotes pointwise multiplication
* make sure that matrices have the same dimensions!
*/
template<class EntryType>
__host__ void basic_matrix_device<EntryType>::mult_itself_pointwise(basic_matrix_device<EntryType>* to_mult){
	//perform matrix arithmetic on cuda device
    mult_pointwise_on_device<<<create_grid_dimension(),create_block_dimension()>>>
                            (d_data, to_mult->d_data, d_data, get_number_of_matrix_entries());
}

/*!
* calculates return = this .* to_mult where '.*' denotes pointwise multiplication
* @return returns a new matrix with the result; there is no memory management bis this class done!
*/
template<class EntryType>
__host__ basic_matrix_device<EntryType>* basic_matrix_device<EntryType>::mult_pointwise(basic_matrix_device<EntryType>* to_mult){
    basic_matrix_device<EntryType>* result = new basic_matrix_device<EntryType>(this, basic_matrix_device<EntryType>::noInit);

	//perform matrix arithmetic on cuda device
    mult_pointwise_on_device<<<create_grid_dimension(),create_block_dimension()>>>
                            (d_data, to_mult->d_data, result->d_data, get_number_of_matrix_entries());

	return result;
}

/*!
* calculates this *= factor
*/
template<class EntryType>
__host__ void basic_matrix_device<EntryType>::scale_itself(CUDA_FLOAT_REAL factor){
    scale_on_device<<<create_grid_dimension(),create_block_dimension()>>>(d_data, factor, d_data, get_number_of_matrix_entries());
}

/*!
* calculates the pointwise inverse of the current matrix
* ATTENTION: zero entries are set to zero!
*/
template<class EntryType>
__host__ void basic_matrix_device<EntryType>::invert_itself_pointwise(){
    invert_pointwise_on_device<<<create_grid_dimension(),create_block_dimension()>>>(d_data, d_data, get_number_of_matrix_entries());
}


/*!
* creates the dimensions of a block for a kernel launch for the current matrix
*/
template<class EntryType>
__host__ dim3 basic_matrix_device<EntryType>::create_block_dimension(){
	dim3 block;
	block.x = MAX_NUMBER_THREADS_PER_BLOCK;
	return block;
}

/*!
* creates the dimensions of a grid for a kernel launch for the current matrix
*/
template<class EntryType>
__host__ dim3 basic_matrix_device<EntryType>::create_grid_dimension(){
    dim3 grid;
    // grid.x = ceil(num / MAX_...)
    grid.x = (get_number_of_matrix_entries() + MAX_NUMBER_THREADS_PER_BLOCK - 1) / MAX_NUMBER_THREADS_PER_BLOCK;

    return grid;
}

template<class EntryType>
__host__ EntryType* basic_matrix_device<EntryType>::get_data(){
    return d_data;
}

template<class EntryType>
__host__ void basic_matrix_device<EntryType>::power_realpart_itself_pointwise(CUDA_FLOAT_REAL exponent){
    power_real_part<<< create_grid_dimension(), create_block_dimension() >>>(get_data(), exponent, get_number_of_matrix_entries());
}


/***********************************************************************************/
/** Compile classes for given template params                                     **/
/***********************************************************************************/
template class basic_matrix_device<CUDA_FLOAT>;
template class basic_matrix_device<CUDA_FLOAT_REAL>;

/***********************************************************************************/
/** Device functions                                                              **/
/***********************************************************************************/
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


// cuda device arithmetic
/*!
* performs result = input + (to_add * factor)
* input == result, add == result is allowed and input == add == result is allowed
*/
#define ENTRY_TYPE CUDA_FLOAT
__global__ void add_mult_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* to_add, CUDA_FLOAT_REAL factor, ENTRY_TYPE* result, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        ENTRY_TYPE input_entry = input[index];
        ENTRY_TYPE add_entry = to_add[index];

        result[index].x =  input_entry.x + (add_entry.x *factor);
        result[index].y =  input_entry.y + (add_entry.y *factor);
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void add_mult_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* to_add, CUDA_FLOAT_REAL factor, ENTRY_TYPE* result, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        ENTRY_TYPE input_entry = input[index];
        ENTRY_TYPE add_entry = to_add[index];

        result[index] =  input_entry + (add_entry * factor);
    }
}
#undef ENTRY_TYPE

/*!
* performs result = input .* to_mult
* input == result, add == result is allowed and input == to_mult == result is allowed
*/
#define ENTRY_TYPE CUDA_FLOAT
__global__ void mult_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* to_mult, ENTRY_TYPE* result, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        ENTRY_TYPE a = input[index];
        ENTRY_TYPE b = to_mult[index];

        result[index].x =  a.x * b.x - a.y * b.y;
        result[index].y =  a.x * b.y + a.y * b.x;
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void mult_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* to_mult, ENTRY_TYPE* result, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        ENTRY_TYPE a = input[index];
        ENTRY_TYPE b = to_mult[index];

        result[index] =  a * b;
    }
}
#undef ENTRY_TYPE

/*!
* performs result = factor * input 
* input == result is allowed
*/
#define ENTRY_TYPE CUDA_FLOAT
__global__ void scale_on_device(ENTRY_TYPE* input, CUDA_FLOAT_REAL factor, ENTRY_TYPE* result, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        ENTRY_TYPE a = input[index];

        result[index].x =  a.x * factor;
        result[index].y =  a.y * factor;
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void scale_on_device(ENTRY_TYPE* input, CUDA_FLOAT_REAL factor, ENTRY_TYPE* result, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        ENTRY_TYPE a = input[index];

        result[index] =  a * factor;
    }
}
#undef ENTRY_TYPE

/*!
* performs pointwise inversion on device in parallel!
* zero entries are set to zero !
* input == result is allowed
*/
#define ENTRY_TYPE CUDA_FLOAT
__global__ void invert_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* result, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        ENTRY_TYPE a = input[index];
        CUDA_FLOAT_REAL a_sq = a.x*a.x + a.y*a.y;

        if(a_sq != 0) {
            result[index].x =  a.x / a_sq;
            result[index].y = -a.y / a_sq;
        }
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void invert_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* result, int number_of_matrix_entries) {
    int index = get_global_index();
    if(index < number_of_matrix_entries) {
        ENTRY_TYPE a = input[index];
        CUDA_FLOAT_REAL a_sq = a*a;

        if(a_sq != 0) {
            result[index] = a / a_sq;
        }
    }
}
#undef ENTRY_TYPE

#define ENTRY_TYPE CUDA_FLOAT
__global__ void init_ident_matrix(ENTRY_TYPE* data, int columns, int rows, int number_of_matrix_entries) {
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries){
        int current_col = 0, current_row = 0, current_matrix = 0;
        get_current_matrix_indices(current_col, current_row, current_matrix, total_index, columns, rows, 1);

        data[total_index].x = (current_col == current_row)?(1.):(0.);
        data[total_index].y = 0.;
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void init_ident_matrix(ENTRY_TYPE* data, int columns, int rows, int number_of_matrix_entries) {
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries){
        int current_col = 0, current_row = 0, current_matrix = 0;
        get_current_matrix_indices(current_col, current_row, current_matrix, total_index, columns, rows, 1);

        data[total_index] = (current_col == current_row)?(1.):(0.);
    }
}
#undef ENTRY_TYPE

#define ENTRY_TYPE CUDA_FLOAT
__global__ void init_zeros(ENTRY_TYPE* data, int number_of_matrix_entries){
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries) {
        data[total_index].x = 0.;
        data[total_index].y = 0.;
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void init_zeros(ENTRY_TYPE* data, int number_of_matrix_entries){
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries) {
        data[total_index] = 0.;
    }
}
#undef ENTRY_TYPE

#define ENTRY_TYPE CUDA_FLOAT
__global__ void init_scalar(ENTRY_TYPE* data, ENTRY_TYPE scalar, int number_of_matrix_entries){
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries) {
        data[total_index].x = scalar.x;
        data[total_index].y = scalar.y;
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void init_scalar(ENTRY_TYPE* data, ENTRY_TYPE scalar, int number_of_matrix_entries){
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries) {
        data[total_index] = scalar;
    }
}
#undef ENTRY_TYPE

#define ENTRY_TYPE CUDA_FLOAT
__global__ void power_real_part(ENTRY_TYPE* data, CUDA_FLOAT_REAL exponent, int number_of_matrix_entries){
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries) {
        data[total_index].x = pow(data[total_index].x, exponent);
        data[total_index].y = 0.;
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void power_real_part(ENTRY_TYPE* data, CUDA_FLOAT_REAL exponent, int number_of_matrix_entries){
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries) {
        data[total_index] = pow(data[total_index], exponent);
    }
}
#undef ENTRY_TYPE

#define ENTRY_TYPE CUDA_FLOAT
__global__ void truncate_2_3(ENTRY_TYPE* data, int columns, int rows, int matrices, int number_of_matrix_entries){
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries) {
        int logic_col = 0, logic_row = 0, logic_matrix = 0;
        get_current_matrix_indices(logic_col, logic_row, logic_matrix, total_index, columns, rows, matrices);

        //maximum number of new entries
        CUDA_FLOAT_REAL factor = (2./3.);
        int N_col = factor * columns;
        int N_row = factor * rows;
        int N_matrix = factor * matrices;

        //get wave numbers inside the current vector
        //...columns
        int wave_number_col = (-1)*(columns - logic_col);
        if(logic_col < (columns/2)+1)
            wave_number_col = logic_col;
        //...rows
        int wave_number_row = (-1)*(rows - logic_row);
        if(logic_row < (rows/2)+1)
            wave_number_row = logic_row;
        //...matrices
        int wave_number_matrix = (-1)*(matrices - logic_matrix);
        if(logic_matrix < (matrices/2)+1)
            wave_number_matrix = logic_matrix;


        //truncate for specific wave numbers
        if(!( abs(wave_number_col) <= N_col/2 && abs(wave_number_row) <= N_row/2 && abs(wave_number_matrix) <= N_matrix/2 )){
            data[total_index].x = 0.;
            data[total_index].y = 0.;
        }
    }
}
#undef ENTRY_TYPE
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ void truncate_2_3(ENTRY_TYPE* data, int columns, int rows, int matrices, int number_of_matrix_entries){
    int total_index = get_global_index();

    if(total_index < number_of_matrix_entries) {
        int logic_col = 0, logic_row = 0, logic_matrix = 0;
        get_current_matrix_indices(logic_col, logic_row, logic_matrix, total_index, columns, rows, matrices);

        //maximum number of new entries
        CUDA_FLOAT_REAL factor = (2./3.);
        int N_col = factor * columns;
        int N_row = factor * rows;
        int N_matrix = factor * matrices;

        //get wave numbers inside the current vector
        //...columns
        int wave_number_col = (-1)*(columns - logic_col);
        if(logic_col < (columns/2)+1)
            wave_number_col = logic_col;
        //...rows
        int wave_number_row = (-1)*(rows - logic_row);
        if(logic_row < (rows/2)+1)
            wave_number_row = logic_row;
        //...matrices
        int wave_number_matrix = (-1)*(matrices - logic_matrix);
        if(logic_matrix < (matrices/2)+1)
            wave_number_matrix = logic_matrix;


        //truncate for specific wave numbers
        if(!( abs(wave_number_col) <= N_col/2 && abs(wave_number_row) <= N_row/2 && abs(wave_number_matrix) <= N_matrix/2 )){
            data[total_index] = 0.;
        }
    }
}
#undef ENTRY_TYPE





