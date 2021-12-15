#include "util.h"
#include <string>
#include <iostream>
#include <sstream>
#include <time.h>
#include <limits>
#include <typeinfo>
#include <math.h>

/***********************************************************************************/
/** Utility functions                                                             **/
/***********************************************************************************/
int iCeil(double x) {
    int Ix = int(x);
    if (x > double(Ix)) return (Ix + 1);
    else return Ix;
}

int iFloor(double x) {
    int Ix = int(x);
    if (Ix >= 0) return Ix;
    if (x == double(Ix)) return Ix;
    return (Ix - 1);
}

int iRound(double x) {
    int Ix = iFloor(x);
    if(x - double(Ix) >= 0.5) return (Ix + 1);
    else return Ix;
}

bool matrix_has_size(matrix_folder* folder, int x_size, int y_size, int z_size) {

    if(folder->get_dimension() != 1) {
        return false;
    } else {
        vector<int> dim = folder->get_matrix(0)->get_matrix_dimension();
        if(dim.size() != 3 || dim[0] != x_size || dim[1] != y_size || dim[2] != z_size){
            return false;
        }
    }
    return true;
}

CUDA_FLOAT_REAL matrix_max(matrix_folder_real* v_device) {
    matrix_host_real* v_x = new matrix_host_real(v_device->get_matrix(0), matrix_host_real::Copy);
    matrix_host_real* v_y = new matrix_host_real(v_device->get_matrix(1), matrix_host_real::Copy);
    matrix_host_real* v_z = new matrix_host_real(v_device->get_matrix(2), matrix_host_real::Copy);
    matrix_host_real_iterator* it_x = matrix_host_real_iterator::create_iterator(v_x);
    matrix_host_real_iterator* it_y = matrix_host_real_iterator::create_iterator(v_y);
    matrix_host_real_iterator* it_z = matrix_host_real_iterator::create_iterator(v_z);

    // Find maximum
    CUDA_FLOAT_REAL max = 0.0;
    while(it_x->has_next() && it_y->has_next() && it_z->has_next()) {
        CUDA_FLOAT_REAL x = it_x->next();
        CUDA_FLOAT_REAL y = it_y->next();
        CUDA_FLOAT_REAL z = it_z->next();
        if(x*x+y*y+z*z > max) {
            max = x*x+y*y+z*z;
        }
    }

    delete v_x;
    delete v_y;
    delete v_z;
    delete it_x;
    delete it_y;
    delete it_z;

    return sqrt(max);
}

void force_mean(matrix_folder* F, CUDA_FLOAT_REAL mean_F) {
    CUDA_FLOAT mean_F_cplx;
    mean_F_cplx.x = mean_F;
    mean_F_cplx.y = 0.0;

    //copy data to device
    cudaMemcpy(F->get_matrix(0)->get_data(), &mean_F_cplx, sizeof(CUDA_FLOAT), cudaMemcpyHostToDevice);
}

bool find_NaN(matrix_folder* F) {
    bool ret = false;
    for(int i = 0; i < F->get_dimension() && !ret; ++i) {
        matrix_host* Mh = new matrix_host(F->get_matrix(i),matrix_host::Copy);
        ret = find_NaN(Mh);
        delete Mh;
    }
    return ret;
}
bool find_NaN(matrix_host* Mh) {
    bool ret = false;
    matrix_host_iterator* it = matrix_host_iterator::create_iterator(Mh);
    while(it->has_next()) {
        CUDA_FLOAT val = it->next();
        if(val.x != val.x || val.y != val.y) {
            ret = true;
            break;
        }
    }
    delete it;
    return ret;
}

void print_matrix(matrix_device* mat){
    int mox = mat->get_matrix_dimension().at(0);
    int moy = mat->get_matrix_dimension().at(1);
    int moz = mat->get_matrix_dimension().at(2);
    CUDA_FLOAT* data_d = mat->get_data();
    CUDA_FLOAT* data_h = new CUDA_FLOAT[mox * moy * moz];
    cudaMemcpy(data_h, data_d, sizeof(CUDA_FLOAT) * mox * moy * moz, cudaMemcpyDeviceToHost);
    DBGOUT("MATRIX PRINT");
    for(int j = 0; j < moy; ++j) {
        for(int i = 0; i < mox; ++i) {
            cout << i << " " << j << " ";
            for(int k = 0; k < moz; ++k) {

                cout << "("<< data_h[i + j * mox + k * mox * moy].x << ","
                           << data_h[i + j * mox + k * mox * moy].y << ") ";
            }
            cout << endl;
        }
    }
    delete [] data_h;
}

void diff_matrix(matrix_device* mat, matrix_device* mat2){
    int mox = mat->get_matrix_dimension().at(0);
    int moy = mat->get_matrix_dimension().at(1);
    int moz = mat->get_matrix_dimension().at(2);
    CUDA_FLOAT* data_d = mat->get_data();
    CUDA_FLOAT* data_h = new CUDA_FLOAT[mox * moy * moz];
    cudaMemcpy(data_h, data_d, sizeof(CUDA_FLOAT) * mox * moy * moz, cudaMemcpyDeviceToHost);
    CUDA_FLOAT* data_d2 = mat2->get_data();
    CUDA_FLOAT* data_h2 = new CUDA_FLOAT[mox * moy * moz];
    cudaMemcpy(data_h2, data_d2, sizeof(CUDA_FLOAT) * mox * moy * moz, cudaMemcpyDeviceToHost);

    DBGOUT("MATRIX DIFF");
    for(int j = 0; j < moy; ++j) {
        for(int i = 0; i < mox; ++i) {
            cout << i << " " << j << " ";
            for(int k = 0; k < moz; ++k) {

                cout << "("<< (data_h[i + j * mox + k * mox * moy].x -data_h2[i + j * mox + k * mox * moy].x) << ","
                           << (data_h[i + j * mox + k * mox * moy].y - data_h2[i + j * mox + k * mox * moy].y) << ") ";
            }
            cout << endl;
        }
    }
    delete [] data_h;
    delete [] data_h2;
}

/***********************************************************************************/
/** Make real functions                                                           **/
/***********************************************************************************/
static __device__ int get_global_index() {
    return (threadIdx.x + (threadIdx.y + (threadIdx.z + (blockIdx.x + (blockIdx.y + (blockIdx.z)
            * gridDim.y) * gridDim.x) * blockDim.z) * blockDim.y) * blockDim.x);
}


/*!
* returns the current index in matrix
* @param total_index is the total current index \in [0;number_of_elements-1]
* @param logic_index i.e. 0=column-, 1=row-, 2=matrix-index is returned
* @param columns the number of columns of the matrix
* @param rows the number of rows
* @param matrices the next higher order index
* @return returns the current index , e.g. if logic_index==1 it will return the current row
*/
static __device__ void get_current_matrix_indices(int& current_col, int& current_row, int& current_matrix,
                                                  int total_index, int columns, int rows, int matrices) {

    int xysize = rows * columns;

    current_col = (total_index % columns);
    current_row = ((total_index % xysize) / columns);
    current_matrix = ((total_index % (xysize * matrices)) / xysize);
}


__global__ void make_real(CUDA_FLOAT* data, int columns, int rows, int matrices){

    //force the signal to be real
    const int total_index = get_global_index();

    if(total_index < columns*rows*matrices) {
        int current_col = 0, current_row = 0, current_matrix = 0;
        get_current_matrix_indices(current_col, current_row, current_matrix, total_index, columns, rows, matrices);

        if(current_col == 0 && current_row == 0) {
            data[total_index].y = 0;
        } else if(current_col == 0) {
            if(current_row < rows/2){
                int update_index = current_col + (rows-current_row)*columns +current_matrix*(columns*rows);
                data[total_index].x =  data[update_index].x;
                data[total_index].y = -data[update_index].y;
            } else if(current_row == rows/2) {
                data[total_index].y = 0;
            }
        } else if(current_row == rows/2 && current_col == columns) {
            data[total_index].y = 0;
        }

    }

}


// Converts first 3 matrices of iteration_vector to real
void make_real_signal(matrix_folder** data){

    for(int i = 0; i < 3; i++) {
        matrix_device*   matrix = data[i]->get_matrix(0);
        int columns    = matrix->get_matrix_size(0);
        int rows       = matrix->get_matrix_size(1);
        int matrices   = matrix->get_matrix_size(2);
        dim3 grid_dim  = matrix->create_grid_dimension();
        dim3 block_dim = matrix->create_block_dimension();
        make_real<<< grid_dim , block_dim >>>( matrix->get_data(), columns, rows, matrices);
    }
}


/***********************************************************************************/
/** Timer class                                                                   **/
/***********************************************************************************/
Timer::Timer() : _Running(false), _Start(clock()), _Tics(0), _Hours(0) {}

int Timer::Start() {
    int ret = _Running?-1:0;
    _Start = clock();
    _Running = true;
    return ret;
}

int Timer::Stop() {
    int ret = _Running?0:-1;
    unsigned long time = clock();
    if(time < _Start) { // wraparound
        _Tics += time + (ULONG_MAX - _Start);
    } else {
        _Tics += time - _Start;
    }
    TicstoHours();
    _Running = false;
    return ret;
}

int Timer::Elapsedms() {
    bool wasRunning = _Running;
    if(wasRunning) Stop();
    int ret = _Hours * 3600000 + _Tics * (1000 / CLOCKS_PER_SEC);
    if(wasRunning) Start();
    return ret;
}

double Timer::Elapseds() {
    bool wasRunning = _Running;
    if(wasRunning) Stop();
    double ret = _Hours * 3600. + double(_Tics) / CLOCKS_PER_SEC;
    if(wasRunning) Start();
    return ret;
}

void Timer::Reset() {
    _Running = false;
    _Start = clock();
    _Tics = 0;
    _Hours = 0;
}

void Timer::TicstoHours() {
    unsigned long TicsperHour = 3600 * CLOCKS_PER_SEC;
    while(_Tics > TicsperHour) {
        _Tics -= TicsperHour;
        _Hours++;
    }
}

