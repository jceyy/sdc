#include "matrix_host.h"
#include "matrix_device.h"

template<class EntryType>
basic_matrix_host<EntryType>::basic_matrix_host() {
	this->data = NULL;
}

template<class EntryType>
basic_matrix_host<EntryType>::basic_matrix_host(EntryType* data_field, std::vector<int> dim) {
	this->init(dim);

	//use data_field as data
	this->data = data_field;
}

/*!
* creates a new host matrix with real part "real_part"
* real_part has to have as many entries as matrix_dim specify!
* the imaginary part is set to zero
*/
/*template<>
basic_matrix_host<CUDA_FLOAT>::basic_matrix_host(double* real_part, std::vector<int> dim){
    this->init(dim);

    //now copy the data
    for(int i = 0; i < this->get_number_of_matrix_entries(); i++) {
        data[i].x = real_part[i];
        data[i].y = 0.0;
    }
}
template<>
basic_matrix_host<CUDA_FLOAT_REAL>::basic_matrix_host(double* real_part, std::vector<int> dim){
    this->init(dim);

    //now copy the data
    for(int i = 0; i < this->get_number_of_matrix_entries(); i++) {
        data[i] = real_part[i];
    }
}*/

template<class EntryType>
basic_matrix_host<EntryType>::basic_matrix_host(basic_matrix_device<EntryType>* matrix, CopyEntries copy){

	this->init(matrix->get_matrix_dimension());


    if(copy == Copy){
		void* target = (void*) this->data;
		void* src = (void*) matrix->get_data();
        cudaMemcpy(target, src, sizeof(EntryType) * this->get_number_of_matrix_entries(), cudaMemcpyDeviceToHost);

        // Wait for transfer to complete before returning from constructor
        cudaDeviceSynchronize();
	}
}

/*!
* creates a new matrix in host memory and sets its entries to zero
* @param matrix_dim specifies the dimensions of the matrix structure of matrix_dim (col, row,..., etc.)
*/
template<class EntryType>
basic_matrix_host<EntryType>::basic_matrix_host(std::vector<int> matrix_dim) {
	//init memory
	this->init(matrix_dim);
	
	//set all entries to zero
	this->init_zeros();
}

template<class EntryType>
void basic_matrix_host<EntryType>::init(std::vector<int> matrix_dim) {
	this->matrix_dimension = matrix_dim;

	//get the number of matrix entries
	int number_of_entries = 1;
    for(size_t i = 0; i < matrix_dim.size(); i++)
		number_of_entries *= matrix_dim[i];

	//allocate memory for data
    this->data = new EntryType[number_of_entries];
}


template<class EntryType>
basic_matrix_host<EntryType>::~basic_matrix_host() {
	if(this->data == NULL)	
		return;

	//free memory
    delete[] this->data;
}

/*!
* returns the total number of matrix entries (with resp. to all dimensions)
*/
template<class EntryType>
int basic_matrix_host<EntryType>::get_number_of_matrix_entries(){
	if(this->matrix_dimension.empty())
		return 0;

	int number_of_entries = 1;
    for(size_t i = 0; i < this->matrix_dimension.size(); i++)
		number_of_entries *= this->matrix_dimension[i];

	return number_of_entries;	
}

/*!
* returns the dimension of the matrix (i.e 1 for a vector, 2 for matrix, etc.)
*/
template<class EntryType>
int basic_matrix_host<EntryType>::get_dimensions() {
	return this->matrix_dimension.size();
}

template<class EntryType>
vector<int> basic_matrix_host<EntryType>::get_matrix_dimensions() {
	return this->matrix_dimension;
}

/*!
* returns the size of the matrix for a given dimension (e.g. for a 2x4 matrix, get_matrix_size(0) = 4 and get_matrix_size(1) = 2)
* if dimension_index is to large it will exit the programm
*/
template<class EntryType>
int basic_matrix_host<EntryType>::get_matrix_size(int dimension_index) {
    if(dimension_index < 0 || dimension_index >= static_cast<int>(this->matrix_dimension.size())) {
        EXIT_ERROR("matrix dimension too low or high");
	}

	return this->matrix_dimension[dimension_index];	
}

template<class EntryType>
void basic_matrix_host<EntryType>::init_zeros(){
    int num_entries = this->get_number_of_matrix_entries();

    memset(this->data, 0, sizeof(EntryType) * num_entries);
    /*for(int i = 0; i < num_entries; i++){
        this->data[i].x = 0.0;
        this->data[i].y = 0.0;
    }*/
}

/*!
* method to get an entry og a host matrix 
* @param indices is a vector which has the indices of the entries, structured like this: (col_index, row_index,...)
*/	
template<class EntryType>
EntryType basic_matrix_host<EntryType>::get_entry(vector<int> indices) {
    int index = this->get_data_index(indices);
    if(index >= this->get_number_of_matrix_entries()) {
        EXIT_ERROR("not able to set entry, dimensions do not agree.");
	}

	return this->data[index];
}

/*!
* method to set an entry to host matrix 
* @param indices is a vector which has the indices of the entries, structured like this: (col_index, row_index,...)
*/
template<class EntryType>
void basic_matrix_host<EntryType>::set_entry(vector<int> indices, EntryType entry){
    int index = this->get_data_index(indices);
    if(index >= this->get_number_of_matrix_entries()) {
        EXIT_ERROR("not able to set entry, dimensions do not agree.");
    }

    this->data[index] = entry;
}

template<>
void basic_matrix_host<CUDA_FLOAT>::set_entry(vector<int> indices, CUDA_FLOAT_REAL entry_x, CUDA_FLOAT_REAL entry_y){
    int index = this->get_data_index(indices);
    if(index >= this->get_number_of_matrix_entries()) {
        EXIT_ERROR("not able to set entry, dimensions do not agree in matrix_host::set_entry(vector<int> indices, CUDA_FLOAT_REAL entry_x, CUDA_FLOAT_REAL entry_y)");
    }

    this->data[index].x = entry_x;
    this->data[index].y = entry_y;
}

/*!
* method to convert an index vector to a global index
* @param indices is a vector which has the indices of the entries, structured like this: (col_index, row_index,...)
*/	
template<class EntryType>
int basic_matrix_host<EntryType>::get_data_index(vector<int> indices) {
	if(indices.size() != this->matrix_dimension.size()){
        EXIT_ERROR("not able to create global index, dimensions do not agree.");
	}

	int index = 0;
	int factor = 1;

    for(size_t i = 0 ; i < indices.size(); i++){
		index += (indices[i]*factor);
		factor *= this->matrix_dimension[i];
	}

	return index;
}


template<class EntryType>
EntryType* basic_matrix_host<EntryType>::get_data(){
	return this->data;
}

/*!
* converts the real part of the current matrix into a double array which is located in host memory
* there is no memory management done by this class for the returned value
*/
template<>
double* basic_matrix_host<CUDA_FLOAT>::real_part_as_double_array(){
    //create result and fill up with data
    double* result = new double[this->get_number_of_matrix_entries()];
    for(int i = 0; i< this->get_number_of_matrix_entries(); i++) {
        result[i] = this->data[i].x;
    }

    return result;
}
template<>
double* basic_matrix_host<CUDA_FLOAT_REAL>::real_part_as_double_array(){
    //create result and fill up with data
    double* result = new double[this->get_number_of_matrix_entries()];
    for(int i = 0; i< this->get_number_of_matrix_entries(); i++) {
        result[i] = this->data[i];
    }

    return result;
}

/*!
* converts the imag part of the current matrix into a double array which is located in host memory
* there is no memory management done by this class for the returned value
*/
template<>
double* basic_matrix_host<CUDA_FLOAT>::imag_part_as_double_array(){
	//create result and fill up with data
	double* result = new double[this->get_number_of_matrix_entries()];
	for(int i = 0; i< this->get_number_of_matrix_entries(); i++) {
		result[i] = this->data[i].y;
	}	

	return result;
}


/***********************************************************************************/
/** Compile classes for given template params                                     **/
/***********************************************************************************/
template class basic_matrix_host<CUDA_FLOAT>;
template class basic_matrix_host<CUDA_FLOAT_REAL>;


