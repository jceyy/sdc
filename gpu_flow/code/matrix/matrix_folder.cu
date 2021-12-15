#include "matrix_folder.h"

template<class EntryType>
__host__ basic_matrix_folder<EntryType>::basic_matrix_folder(int dimension) {
	this->dim = dimension;

	//allocate memory
    this->matrices = new basic_matrix_device<EntryType>*[dimension];
	//...init all to NULL
	for(int i = 0 ; i < dimension; i++)
		this->matrices[i] = NULL;
	
}

/*!
* frees the memory for the current folder and all its stored!!!! device matrices
*
*/
template<class EntryType>
__host__ basic_matrix_folder<EntryType>::~basic_matrix_folder(){
	//free all matrices
	for(int i = 0 ; i < this->dim; i++) {
		if(matrices[i] != NULL)
			delete matrices[i];
	}
	
}

/*!
* adds a new matrix to the folder
* @param index is the position where m can be found later, if there is already an old matrix at this location memory of the older one will be freed and pointer will be overwritten
* @param m the new matrix to be added, stays the whole time under the memory management of the folder
*/
template<class EntryType>
__host__ void basic_matrix_folder<EntryType>::add_matrix(int index, basic_matrix_device<EntryType>* m){
	if(this->matrices[index] != NULL)
		delete matrices[index];

	this->matrices[index] = m;
}

/*!
* returns a pointer to the matrix located at index index or null if none is there
* ATTENTION: memory is managed by the folder!!!
*/
template<class EntryType>
__host__ basic_matrix_device<EntryType>* basic_matrix_folder<EntryType>::get_matrix(int index){
	return this->matrices[index];
}


/*!
* calculates return = this + (to_add * factor)
* @return returns a new matrix folder; there is no memory management for return value by this class
*/
template<class EntryType>
__host__ basic_matrix_folder<EntryType>* basic_matrix_folder<EntryType>::add_mult_pointwise(basic_matrix_folder<EntryType>* to_add, CUDA_FLOAT_REAL factor){
	//get minimum dimension
	int min_dim = this->dim;
	if(this->dim > to_add->dim)
		min_dim = to_add->dim;

	//iterate over matrices and to add_mult for each matrix and add it to the new folder
    basic_matrix_folder<EntryType>* return_folder = new basic_matrix_folder<EntryType>(min_dim);
	for(int i = 0; i < min_dim; i++) {
        basic_matrix_device<EntryType>* current_matrix = this->get_matrix(i);
        basic_matrix_device<EntryType>* return_matrix = current_matrix->add_mult_pointwise(to_add->get_matrix(i), factor);
		return_folder->add_matrix(i, return_matrix);
	}
	return return_folder;	
}

/*!
* calculates this = this + (to_add*factor)
*/
template<class EntryType>
__host__ void basic_matrix_folder<EntryType>::add_mult_itself_pointwise(basic_matrix_folder<EntryType>* to_add, CUDA_FLOAT_REAL factor){
	//check if matrix dimensions are right
    if(to_add->dim != this->dim){
        EXIT_ERROR("matrices to not have the same dimensions!");
    }

	//iterate over the matrices and add them
	for(int i = 0; i < this->dim; i++) {
        basic_matrix_device<EntryType>* current_matrix = this->get_matrix(i);
		current_matrix->add_mult_itself_pointwise(to_add->get_matrix(i), factor);
	}
}

/*!
* calculates this = this + (to_add*factor) but only for the given index
*/
template<class EntryType>
__host__ void basic_matrix_folder<EntryType>::add_mult_itself_pointwise_specific_index(basic_matrix_device<EntryType>* to_add, CUDA_FLOAT_REAL factor, int index){
	//check if matrix dimensions are right
	#ifdef DEBUG
	std::vector<int> ref_size = this->get_matrix(index)->get_matrix_dimension();
	std::vector<int> size = to_add->get_matrix_dimension();
	if(size.size() != ref_size.size()){
        EXIT_ERROR("matrices do not have the same dimensions!");
	}
	for(int i = 0; i < size.size(); i++){
		if(size[i] != ref_size[i]){
            EXIT_ERROR("matrices do not have the same dimensions!");
		}
	}
	#endif

	this->get_matrix(index)->add_mult_itself_pointwise(to_add, factor);
}

/*!
* calculates this = this .* to_mult (pointwise for each matrix in the current folder)
*/
template<class EntryType>
__host__ void basic_matrix_folder<EntryType>::mult_itself_pointwise(basic_matrix_folder<EntryType>* to_mult) {
	//check if matrix dimensions are right
	#ifdef DEBUG
	if(to_mult->dim < this->dim){
        EXIT_ERROR("matrices to not have the same dimensions!");
	}
	#endif

	for(int i = 0; i < this->dim; i++) {
        basic_matrix_device<EntryType>* current_matrix = this->get_matrix(i);
		current_matrix->mult_itself_pointwise(to_mult->get_matrix(i));
	}
}

/*!
* calculates return = this .* to_mult (pointwise for each matrix in the current folder)
* there is no memory management done by this class for the return value
*/
template<class EntryType>
__host__ basic_matrix_folder<EntryType>* basic_matrix_folder<EntryType>::mult_pointwise(basic_matrix_folder<EntryType>* to_mult){
	//get minimum dimension
	int min_dim = this->dim;
	if(this->dim > to_mult->dim)
		min_dim = to_mult->dim;

	//iterate over matrices and to add_mult for each matrix and add it to the new folder
    basic_matrix_folder<EntryType>* return_folder = new basic_matrix_folder<EntryType>(min_dim);
	for(int i = 0; i < min_dim; i++) {
        basic_matrix_device<EntryType>* current_matrix = this->get_matrix(i);
        basic_matrix_device<EntryType>* return_matrix = current_matrix->mult_pointwise(to_mult->get_matrix(i));
		return_folder->add_matrix(i, return_matrix);
	}
	return return_folder;	
}

/*!
* scales all entries of the current folder with factor
*/
template<class EntryType>
__host__ void basic_matrix_folder<EntryType>::scale_itself(CUDA_FLOAT_REAL factor){
	//iterate over the matrices and add them
    for(int i = 0; i < this->dim; i++) {
        this->get_matrix(i)->scale_itself(factor);
	}
}

/*!
* inverts all entries of the current matrix pointwise
*/
template<class EntryType>
__host__ void basic_matrix_folder<EntryType>::invert_itself_pointwise(){
	//iterate over the matrices and add them
	for(int i = 0; i < this->dim; i++) {
        this->get_matrix(i)->invert_itself_pointwise();
	}
}

template<class EntryType>
__host__ int basic_matrix_folder<EntryType>::get_dimension(){
	return this->dim;
}

/***********************************************************************************/
/** Compile classes for given template params                                     **/
/***********************************************************************************/
template class basic_matrix_folder<CUDA_FLOAT>;
template class basic_matrix_folder<CUDA_FLOAT_REAL>;






