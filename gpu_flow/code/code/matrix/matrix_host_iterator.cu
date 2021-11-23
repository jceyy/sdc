#include "matrix_host_iterator.h"

template<class EntryType>
basic_matrix_host_iterator<EntryType>::basic_matrix_host_iterator() {
	this->current_index = -1;
}


template<class EntryType>
basic_matrix_host_iterator<EntryType>*  basic_matrix_host_iterator<EntryType>::create_iterator(basic_matrix_host<EntryType>* matrix) {
    basic_matrix_host_iterator<EntryType>* it = new basic_matrix_host_iterator<EntryType>();

	it->data = matrix;

	return it;
}

template<class EntryType>
bool basic_matrix_host_iterator<EntryType>::has_next() {
	if(current_index < data->get_number_of_matrix_entries()-1)
		return true;
	else
		return false;
}


template<class EntryType>
EntryType basic_matrix_host_iterator<EntryType>::next(){
	this->current_index++;
    EntryType* d = data->get_data();
	return d[this->current_index];
}

/*!
* returns the matrix indices for the current entry
* @return returns a vector with indices ordered like this: (column, row, etc.)
*/
template<class EntryType>
std::vector<int> basic_matrix_host_iterator<EntryType>::get_current_indices() {
	std::vector<int> indices;

	std::vector<int> d_dim = this->data->get_matrix_dimensions();
	int mod = 1;
	int div = 1;
    for(size_t i = 0; i < d_dim.size(); i++) {
		mod *= d_dim.at(i);
		if(i > 0)
			div *= d_dim.at(i-1);

		int matrix_index = (this->current_index % mod) / div;
		
		indices.push_back(matrix_index);
	}

	return indices;	
}

/***********************************************************************************/
/** Compile classes for given template params                                     **/
/***********************************************************************************/
template class basic_matrix_host_iterator<CUDA_FLOAT>;
template class basic_matrix_host_iterator<CUDA_FLOAT_REAL>;

