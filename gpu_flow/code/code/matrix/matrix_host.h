#ifndef MATRIX_HOST_H
#define MATRIX_HOST_H

//system includes
#include <cuda.h>
#include <iostream>
#include <vector>
using namespace std;

//my includes
#include "../cuda_defines.h"

template<class EntryType>
class basic_matrix_device;


template<class EntryType>
class basic_matrix_host {

private:
    basic_matrix_host();
	void init(std::vector<int> matrix_dim);
	

protected:
	//vector keeps information about the dimensions of the matrix
	vector<int> matrix_dimension;

	//data field
    EntryType* data;

	//convert indices to a global index
	int get_data_index(vector<int> indices);
	
public:
    enum CopyEntries { noCopy, Copy };
    basic_matrix_host(vector<int> matrix_dim);
    basic_matrix_host(EntryType* data_field, vector<int> matrix_dim);
    //basic_matrix_host(double* real_part, vector<int> matrix_dim);//does not work if EntryType=CUDA_FLOAT_REAL=cufftDouble=double !
    basic_matrix_host(basic_matrix_device<EntryType>* matrix, CopyEntries copy);
    ~basic_matrix_host();

	//to get info about matrix_host
	int get_number_of_matrix_entries();
	int get_dimensions();
    vector<int> get_matrix_dimensions();
	int get_matrix_size(int dimension_index);
    EntryType* get_data();
	
	//acces entries
	void init_zeros();	
    EntryType get_entry(vector<int> indices);
    void set_entry(vector<int> indices, EntryType entry);
    void set_entry(vector<int> indices, CUDA_FLOAT_REAL entry_x, CUDA_FLOAT_REAL entry_y);
	
	//to convert information
	double* real_part_as_double_array();
    double* imag_part_as_double_array();
	
};

typedef basic_matrix_host<CUDA_FLOAT> matrix_host;
typedef basic_matrix_host<CUDA_FLOAT_REAL> matrix_host_real;

#endif






