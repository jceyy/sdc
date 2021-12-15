#ifndef MATRIX_HOST_ITERATOR_H
#define MATRIX_HOST_ITERATOR_H

//system includes
#include <cuda.h>
#include <vector>

//my includes
#include "../cuda_defines.h"
#include "matrix_host.h"

template<class EntryType>
class basic_matrix_host_iterator {

private:
    basic_matrix_host_iterator();
	int current_index;
	

protected:
    basic_matrix_host<EntryType>* data;
	
public:
	/*!
	* creates an iterator on matrix_host matrix, make sure matrix exists as long as the created iterator exists, no memory management is done by this class for matrix!
	* @param matrix the matrix on which the iterator is iterating, NULL is not allowed
	* @return returns a pointer to an iterator , there is no memory management done!
	*/
    static basic_matrix_host_iterator*  create_iterator(basic_matrix_host<EntryType>* matrix);

	bool has_next();
    EntryType next();
    vector<int> get_current_indices();


};

typedef basic_matrix_host_iterator<CUDA_FLOAT> matrix_host_iterator;
typedef basic_matrix_host_iterator<CUDA_FLOAT_REAL> matrix_host_real_iterator;


#endif

