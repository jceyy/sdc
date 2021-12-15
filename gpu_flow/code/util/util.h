#ifndef UTIL_H
#define UTIL_H
#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"
#include "../matrix/matrix_host.h"
#include "../matrix/matrix_host_iterator.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>

/***********************************************************************************/
/** Utility functions                                                             **/
/***********************************************************************************/
int iCeil(double x);
int iFloor(double x);
int iRound(double x);
bool matrix_has_size(matrix_folder* folder, int x_size, int y_size, int z_size);
CUDA_FLOAT_REAL matrix_max(matrix_folder_real* v_device);

void force_mean(matrix_folder* F, CUDA_FLOAT_REAL mean_F);
bool find_NaN(matrix_folder* F);
bool find_NaN(matrix_host* F);

void print_matrix(matrix_device* mat);
void diff_matrix(matrix_device* mat, matrix_device* mat2);


/***********************************************************************************/
/** Device functions                                                              **/
/***********************************************************************************/
static __device__ int get_global_index();
static __device__ void get_current_matrix_indices(int &current_col, int &current_row, int &current_matrix, int total_index, int columns, int rows, int matrices);

/***********************************************************************************/
/** Make real functions                                                           **/
/***********************************************************************************/
void make_real_signal(matrix_folder** data);
static __global__ void make_real(CUDA_FLOAT* data, int columns, int rows, int matrices);



/***********************************************************************************/
/** Timer class                                                                   **/
/***********************************************************************************/
class Timer {
public:
    Timer();
    int Start();
    int Stop();
    void Reset();
    int Elapsedms();
    double Elapseds();
private:
    void TicstoHours();

    bool _Running;
    unsigned long _Start;
    unsigned long _Tics;
    unsigned long _Hours;
};
#endif // UTIL_H
