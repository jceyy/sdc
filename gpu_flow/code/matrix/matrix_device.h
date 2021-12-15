#ifndef MATRIX_CUDA_DEVICE_H
#define MATRIX_CUDA_DEVICE_H

//system includes
#include <cuda.h>
#include <vector>
#include <cstdio>

//my includes
#include "../cuda_defines.h"
#include "matrix_host.h"

template<class EntryType>
class basic_matrix_device {

private:
    __host__ basic_matrix_device();
    __host__ void init(basic_matrix_host<EntryType>* m);
    __host__ void init(basic_matrix_device<EntryType>* m);

protected:
	//matrix data stored on CUDA device
    EntryType* d_data;

	//dimension of the matrix, stored on host
	std::vector<int> dimension;
		
public:
    enum InitEntries { noInit, Copy };

	//for construction of device matrices
    __host__ basic_matrix_device(vector<int> dimensions);
    __host__ basic_matrix_device(basic_matrix_device<EntryType>* m, InitEntries initEntries);
    __host__ basic_matrix_device(basic_matrix_host<EntryType>* m, InitEntries initEntries);
    __host__ ~basic_matrix_device();

    //for setting values
    __host__ void set_zero();
    __host__ void set_identity();
    __host__ void set_scalar(EntryType val);

	//to check dimensions of matrix
    __host__ std::vector<int> get_matrix_dimension() const;
    __host__ int get_matrix_size(int dimension_index);
	__host__ int get_number_of_matrix_entries();

	//access to the data field
    __host__ EntryType* get_data();

	//arithmetic operations
    __host__ basic_matrix_device<EntryType>* add_mult_pointwise(basic_matrix_device<EntryType>* to_add, CUDA_FLOAT_REAL factor);
    __host__ void add_mult_itself_pointwise(basic_matrix_device<EntryType>* to_add, CUDA_FLOAT_REAL factor);
    __host__ void mult_itself_pointwise(basic_matrix_device<EntryType>* to_mult);
    __host__ basic_matrix_device<EntryType>* mult_pointwise(basic_matrix_device<EntryType>* to_mult);
	__host__ void scale_itself(CUDA_FLOAT_REAL factor);
	__host__ void invert_itself_pointwise();
    __host__ void power_realpart_itself_pointwise(CUDA_FLOAT_REAL exponent);
	
	//create grid and block dimensionality
	__host__ dim3 create_block_dimension();
	__host__ dim3 create_grid_dimension();
	
};

typedef basic_matrix_device<CUDA_FLOAT> matrix_device;
typedef basic_matrix_device<CUDA_FLOAT_REAL> matrix_device_real;

//helper functions
__device__ static int get_global_index();
//to create column, row and matrix index from a global index
__device__ static void get_current_matrix_indices(int &current_col, int &current_row, int &current_matrix,
                                                  int total_index, int columns, int rows, int matrices);

// cuda device arithmetic
// functions for complex numbers
#define ENTRY_TYPE CUDA_FLOAT
__global__ static void add_mult_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* to_add, CUDA_FLOAT_REAL factor, ENTRY_TYPE* result, int number_of_matrix_entries);
__global__ static void mult_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* to_mult, ENTRY_TYPE* result, int number_of_matrix_entries);
__global__ static void scale_on_device(ENTRY_TYPE* input, CUDA_FLOAT_REAL factor, ENTRY_TYPE* result, int number_of_matrix_entries);
__global__ static void invert_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* result, int number_of_matrix_entries);

//sets all entries of data to zero
__global__ static void init_zeros(ENTRY_TYPE* data, int number_of_matrix_entries);
__global__ static void init_scalar(ENTRY_TYPE* data, ENTRY_TYPE scalar, int number_of_matrix_entries);
__global__ static void init_ident_matrix(ENTRY_TYPE* data, int columns, int rows, int number_of_matrix_entries);

//sets all imag parts to zero and raises all real parts to the power
__global__ static void power_real_part(ENTRY_TYPE* data, CUDA_FLOAT_REAL exponent, int number_of__matrix_entries);
#undef ENTRY_TYPE

// functions for real numbers
#define ENTRY_TYPE CUDA_FLOAT_REAL
__global__ static void add_mult_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* to_add, CUDA_FLOAT_REAL factor, ENTRY_TYPE* result, int number_of_matrix_entries);
__global__ static void mult_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* to_mult, ENTRY_TYPE* result, int number_of_matrix_entries);
__global__ static void scale_on_device(ENTRY_TYPE* input, CUDA_FLOAT_REAL factor, ENTRY_TYPE* result, int number_of_matrix_entries);
__global__ static void invert_pointwise_on_device(ENTRY_TYPE* input, ENTRY_TYPE* result, int number_of_matrix_entries);

//sets all entries of data to zero
__global__ static void init_zeros(ENTRY_TYPE* data, int number_of_matrix_entries);
__global__ static void init_scalar(ENTRY_TYPE* data, ENTRY_TYPE scalar, int number_of_matrix_entries);
__global__ static void init_ident_matrix(ENTRY_TYPE* data, int columns, int rows, int number_of_matrix_entries);

//sets all imag parts to zero and raises all real parts to the power
__global__ static void power_real_part(ENTRY_TYPE* data, CUDA_FLOAT_REAL exponent, int number_of__matrix_entries);
#undef ENTRY_TYPE

#endif



