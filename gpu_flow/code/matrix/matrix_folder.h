#ifndef MATRIX_DEVICE_MATRIX_FOLDER_H
#define MATRIX_DEVICE_MATRIX_FOLDER_H

//system includes 
#include <cuda.h>

//my includes
#include "../cuda_defines.h"
#include "matrix_device.h"

template<class EntryType>
class basic_matrix_folder {

protected:
	//stored on host
	int dim;

	//stored on cuda device
    basic_matrix_device<EntryType>** matrices;


public: 
    __host__ basic_matrix_folder(int dim);
    __host__ ~basic_matrix_folder();

	//to get information about folder
    __host__ void add_matrix(int index, basic_matrix_device<EntryType>* m);
    __host__ basic_matrix_device<EntryType>* get_matrix(int index);
	__host__ int get_dimension();
	


	//some arithmetic operations
    __host__ basic_matrix_folder<EntryType>* add_mult_pointwise(basic_matrix_folder<EntryType>* to_add, CUDA_FLOAT_REAL factor);
    __host__ void add_mult_itself_pointwise(basic_matrix_folder<EntryType>* to_add, CUDA_FLOAT_REAL factor);
    __host__ void add_mult_itself_pointwise_specific_index(basic_matrix_device<EntryType>* to_add, CUDA_FLOAT_REAL factor, int index);
	__host__ void scale_itself(CUDA_FLOAT_REAL factor);
    __host__ void mult_itself_pointwise(basic_matrix_folder<EntryType>* to_mult);
    __host__ basic_matrix_folder<EntryType>* mult_pointwise(basic_matrix_folder<EntryType>* to_mult);
	__host__ void invert_itself_pointwise();
};

typedef basic_matrix_folder<CUDA_FLOAT> matrix_folder;
typedef basic_matrix_folder<CUDA_FLOAT_REAL> matrix_folder_real;


#endif


