#include "init.h"

void matrix_device_init::init_zeros(matrix_device* to_init) {

    to_init->set_zero();
}


void matrix_device_init::init_random(matrix_device* to_init, int seed, CUDA_FLOAT_REAL max_amplitude) {

    srand(seed);

    DBGSYNC();
    int span = to_init->get_matrix_size(0);
    int total_size = to_init->get_matrix_size(1) * to_init->get_matrix_size(2);
    init_random_on_GPU<<<create_block_dim(total_size), create_grid_dim(total_size)>>>
                      (total_size, to_init->get_data(), span, seed, max_amplitude);

    DBGSYNC();
}

void matrix_device_init::init_dislocation(matrix_device* theta, int x_mode, int y_mode, CUDA_FLOAT_REAL max_amplitude) {

    int num_x_real = 2 * (theta->get_matrix_size(0) - 1);
    int num_y_real = theta->get_matrix_size(1);
    CUDA_FLOAT_REAL* real_data = new CUDA_FLOAT_REAL[num_x_real*num_y_real];

    CUDA_FLOAT_REAL delta_x = 1.0 / num_x_real;
    CUDA_FLOAT_REAL delta_y = 1.0 / num_y_real;
    CUDA_FLOAT_REAL x_point_1 = (num_x_real/2.0)*delta_x + (delta_x/2.0);
    CUDA_FLOAT_REAL x_point_2 = (num_x_real/2.0)*delta_x + (delta_x/2.0);
    CUDA_FLOAT_REAL y_point_1 = (num_y_real/4.0)*delta_y + (delta_y/2.0);
    CUDA_FLOAT_REAL y_point_2 = (3.0*num_y_real/4.0)*delta_y + (delta_y/2.0);

    for(int j = 0; j < num_y_real; j++){

        for(int i = 0; i < num_x_real; i++){
            CUDA_FLOAT_REAL current_x = i*delta_x;
            CUDA_FLOAT_REAL current_y = j*delta_y;
            CUDA_FLOAT_REAL difference_x_1 = i*delta_x - x_point_1;
            CUDA_FLOAT_REAL difference_x_2 = i*delta_x - x_point_2;
            CUDA_FLOAT_REAL difference_y_1 = j*delta_y - y_point_1;
            CUDA_FLOAT_REAL difference_y_2 = j*delta_y - y_point_2;

            CUDA_FLOAT_REAL phase_1 = atan2(difference_y_1, difference_x_1);
            CUDA_FLOAT_REAL phase_2 = atan2(difference_y_2, difference_x_2);

            real_data[i + j*num_x_real] = max_amplitude * cos(  x_mode*2.0*M_PI * current_x
                                                                + y_mode*2.0*M_PI * current_y
                                                                - phase_1 + phase_2);
        }
    }

    //transfer to GPU to apply cufft
    CUDA_FLOAT_REAL* real_data_device;
    cudaMalloc((void**) &real_data_device, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real);
    cudaMemcpy(real_data_device, real_data, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real, cudaMemcpyHostToDevice);

    /*add_random_on_GPU_real<<<create_block_dim(num_x_real*num_y_real), create_grid_dim(num_x_real*num_y_real)>>>
                     (num_x_real*num_y_real, real_data_device, 0, max_amplitude * 1e-1);*/

    //apply real to complex fft to it
    fft(real_data_device, theta->get_data(), num_x_real, num_y_real);

    cudaFree(real_data_device);
    delete [] real_data;

}

void matrix_device_init::init_rectangle(matrix_device* theta, int x_mode, int y_mode, CUDA_FLOAT_REAL max_amplitude) {
    int num_x_real = 2*(theta->get_matrix_size(0) - 1);
    int num_y_real = (theta->get_matrix_size(1));
    CUDA_FLOAT_REAL* real_data = new CUDA_FLOAT_REAL[num_x_real*num_y_real];

    double delta_x = double(x_mode)/double(num_x_real);
    double delta_y = double(y_mode)/double(num_y_real);

    for(int j = 0; j < num_y_real; j++){
        for(int i = 0; i < num_x_real; i++){

            if(i < num_x_real / 8) {
                real_data[i + j*num_x_real] = ((rand() - ((double) (RAND_MAX))/2.0)/(((double) (RAND_MAX))/2.0)) * max_amplitude;
            } else {
                real_data[i + j*num_x_real] = max_amplitude * cos(2.0*M_PI*(i*delta_x + j*delta_y));
            }
        }
    }

    //transfer to GPU to apply cufft
    CUDA_FLOAT_REAL* real_data_device;
    cudaMalloc((void**) &real_data_device, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real);
    cudaMemcpy(real_data_device, real_data, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real, cudaMemcpyHostToDevice);

    //apply real to complex fft to it
    fft(real_data_device, theta->get_data(), num_x_real, num_y_real);

    cudaFree(real_data_device);
    delete [] real_data;

}

void matrix_device_init::init_isr(matrix_device* theta, int x_mode, int y_mode, CUDA_FLOAT_REAL max_amplitude) {
    int num_x_real = 2*(theta->get_matrix_size(0) - 1);
    int num_y_real = (theta->get_matrix_size(1));
    CUDA_FLOAT_REAL* real_data = new CUDA_FLOAT_REAL[num_x_real*num_y_real];

    double phase = (2*M_PI*rand()) / ((double) (RAND_MAX));
    double delta_x = double(x_mode)/double(num_x_real);
    double delta_y = double(y_mode)/double(num_y_real);

    for(int j = 0; j < num_y_real; j++){
        for(int i = 0; i < num_x_real; i++){
            real_data[i + j*num_x_real] = max_amplitude * cos(2.0*M_PI*(i*delta_x + j*delta_y) + phase);
        }
    }

    //transfer to GPU to apply cufft
    CUDA_FLOAT_REAL* real_data_device;
    cudaMalloc((void**) &real_data_device, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real);
    cudaMemcpy(real_data_device, real_data, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real, cudaMemcpyHostToDevice);


    //apply real to complex fft to it
    fft(real_data_device, theta->get_data(), num_x_real, num_y_real);

    cudaFree(real_data_device);
    delete [] real_data;

}

void matrix_device_init::init_sv(matrix_device* theta, int y_mode, double x_frac, CUDA_FLOAT_REAL max_amplitude) {
    int num_x_real = 2*(theta->get_matrix_size(0) - 1);
    int num_y_real = (theta->get_matrix_size(1));
    CUDA_FLOAT_REAL* real_data = new CUDA_FLOAT_REAL[num_x_real*num_y_real];

    double delta_y = double(y_mode)/double(num_y_real);
    double x_left  = 0.5 * (1.0 - x_frac);
    double x_right = 1.0 - x_left;

    for(int j = 0; j < num_y_real; j++){
        for(int i = 0; i < num_x_real; i++){
            double x = i/double(num_x_real);
            double y = j/double(num_y_real);

            double phase = -0.5*sin(2*M_PI*y) - x*x;
            if(x > x_right) phase += 1.0;
            else if(x > x_left) phase += 0.5*(1.0 - cos(M_PI*(x-x_left)/x_frac));

            real_data[i + j*num_x_real] = max_amplitude * cos(2.0*M_PI*(j*delta_y + phase));
        }
    }

    //transfer to GPU to apply cufft
    CUDA_FLOAT_REAL* real_data_device;
    cudaMalloc((void**) &real_data_device, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real);
    cudaMemcpy(real_data_device, real_data, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real, cudaMemcpyHostToDevice);


    //apply real to complex fft to it
    fft(real_data_device, theta->get_data(), num_x_real, num_y_real);

    cudaFree(real_data_device);
    delete [] real_data;

}

void matrix_device_init::init_cr(matrix_device* theta, int y_mode, double x_frac, CUDA_FLOAT_REAL max_amplitude) {

    // Determine system size in real space, reserve on CPU
    const int num_x_real = 2 * (theta->get_matrix_size(0) - 1);
    const int num_y_real = theta->get_matrix_size(1);
    CUDA_FLOAT_REAL* real_data = new CUDA_FLOAT_REAL[num_x_real*num_y_real];

    // Locations of dislocations
    const CUDA_FLOAT_REAL delta_x = 1.0 / num_x_real;
    const CUDA_FLOAT_REAL delta_y = 1.0 / num_y_real;
    const CUDA_FLOAT_REAL x_point_1 = 0.5 * ((1.0 - x_frac) * num_x_real + 1) * delta_x;
    const CUDA_FLOAT_REAL x_point_2 = 1.0 - x_point_1;
    const CUDA_FLOAT_REAL y_point_1 = 0.5 * (num_y_real + 1) * delta_y;
    const CUDA_FLOAT_REAL y_point_2 = y_point_1;

    for(int j = 0; j < num_y_real; j++) {
        for(int i = 0; i < num_x_real; i++) {

            CUDA_FLOAT_REAL current_x = i*delta_x;
            CUDA_FLOAT_REAL current_y = j*delta_y;
            CUDA_FLOAT_REAL difference_x_1 = current_x - x_point_1;
            CUDA_FLOAT_REAL difference_y_1 = current_y - y_point_1;
            CUDA_FLOAT_REAL difference_x_2 = current_x - x_point_2;
            CUDA_FLOAT_REAL difference_y_2 = current_y - y_point_2;

            CUDA_FLOAT_REAL phase_1 = atan2(difference_y_1, difference_x_1);
            CUDA_FLOAT_REAL phase_2 = atan2(difference_y_2, difference_x_2);

            real_data[i + j*num_x_real] = max_amplitude * cos(  y_mode*2.0*M_PI * current_y
                                                                - phase_1 + phase_2);
        }
    }

    //transfer to GPU to apply cufft
    CUDA_FLOAT_REAL* real_data_device;
    cudaMalloc((void**) &real_data_device, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real);
    cudaMemcpy(real_data_device, real_data, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real, cudaMemcpyHostToDevice);

    /*add_random_on_GPU_real<<<create_block_dim(num_x_real*num_y_real), create_grid_dim(num_x_real*num_y_real)>>>
                     (num_x_real*num_y_real, real_data_device, 0, max_amplitude * 1e-1);*/

    //apply real to complex fft to it
    fft(real_data_device, theta->get_data(), num_x_real, num_y_real);

    cudaFree(real_data_device);
    delete [] real_data;
}

__global__ void init_giantspiral_device(CUDA_FLOAT_REAL* real_data, int num_x_real, int num_y_real, int mode, CUDA_FLOAT_REAL max_amplitude) {
    int index = get_global_index();
    if(index < num_x_real*num_y_real) {
        const CUDA_FLOAT_REAL delta_x = 1.0 / num_x_real;
        const CUDA_FLOAT_REAL delta_y = 1.0 / num_y_real;
        const CUDA_FLOAT_REAL x_point_1 = 0.5;
        const CUDA_FLOAT_REAL y_point_1 = 0.5;
        const CUDA_FLOAT_REAL x_point_2 = 0.8;
        const CUDA_FLOAT_REAL y_point_2 = 0.5;

        const int i = index % num_x_real;
        const int j = index / num_x_real;

        CUDA_FLOAT_REAL current_x = i*delta_x;
        CUDA_FLOAT_REAL current_y = j*delta_y;
        CUDA_FLOAT_REAL difference_x_1 = current_x - x_point_1;
        CUDA_FLOAT_REAL difference_y_1 = current_y - y_point_1;
        CUDA_FLOAT_REAL difference_x_2 = current_x - x_point_2;
        CUDA_FLOAT_REAL difference_y_2 = current_y - y_point_2;

        CUDA_FLOAT_REAL r = sqrt(difference_x_1 * difference_x_1 + difference_y_1 * difference_y_1);

        CUDA_FLOAT_REAL phase_1 = atan2(difference_y_1, difference_x_1);
        CUDA_FLOAT_REAL phase_2 = atan2(difference_y_2, difference_x_2);

        real_data[index] = max_amplitude * cos(2.0*M_PI * mode * r - phase_1 + phase_2);
    }
}

void matrix_device_init::init_giantspiral(matrix_device *theta, int mode, CUDA_FLOAT_REAL max_amplitude)
{
    //DBGOUT("! CHANGED INIT GIANTSPIRAL");
    // Determine system size in real space, reserve on CPU
    const int num_x_real = 2 * (theta->get_matrix_size(0) - 1);
    const int num_y_real = theta->get_matrix_size(1);

    //transfer to GPU to apply cufft
    CUDA_FLOAT_REAL* real_data_device;
    cudaMalloc((void**) &real_data_device, sizeof(CUDA_FLOAT_REAL) * num_x_real * num_y_real);
    init_giantspiral_device<<<create_block_dim(num_x_real*num_y_real), create_grid_dim(num_x_real*num_y_real)>>>
                           (real_data_device, num_x_real, num_y_real, mode, max_amplitude);

    //apply real to complex fft to it
    fft(real_data_device, theta->get_data(), num_x_real, num_y_real);

    cudaFree(real_data_device);
}

__global__ void init_test_device(CUDA_FLOAT* f, CUDA_FLOAT* g, CUDA_FLOAT* theta, CUDA_FLOAT* F, CUDA_FLOAT* G, int mox, int moy, int moz)
{
    int index = get_global_index();
    if(index < mox*moy*moz) {
        int x = index % mox;
        int y = (index / mox) % moy;
        int z = index / (mox*moy);

        CUDA_FLOAT val;
        CUDA_FLOAT_REAL amp = 1e-6;
        // Set f
        //val.x = (y==0 && x!=0)?(1+x*(x+z)):0;
        val.x = (y==0 && x!=0)?(amp*sin(CUDA_FLOAT_REAL(x))):0;
        val.y = 0;
        f[index] = val;

        // Set g
        val.x = 0;
        val.y = 0;
        g[index] = val;

        // Set theta
        //val.x = (y==0 && x!=0)?(1+x*(x+z)):0;
        val.x = (y==0 && x!=0)?(amp*sin(CUDA_FLOAT_REAL(x))):0;
        val.y = 0;
        theta[index] = val;

        if(x==0&&y==0) {
            //val.x = (z+1);
            val.x = (amp*cos(CUDA_FLOAT_REAL(z)));
            val.y = 0;
            F[z] = val;

            val.x = 0;
            val.y = 0;
            G[z] = val;
        }
    }
}

void matrix_device_init::init_test(matrix_device *f, matrix_device *g, matrix_device *theta, matrix_device *F, matrix_device *G)
{
    int mox = theta->get_matrix_size(0);
    int moy = theta->get_matrix_size(1);
    int moz = theta->get_matrix_size(2);
    init_test_device<<<create_block_dim(mox*moy*moz), create_grid_dim(mox*moy*moz)>>>
            (f->get_data(),g->get_data(),theta->get_data(),F->get_data(),G->get_data(),mox,moy,moz);
}

void matrix_device_init::fft(CUDA_FLOAT_REAL* real_data_device_in, CUDA_FLOAT *data_device_out,
                             int num_x_real, int num_y_real)
{
    // Execute fft
    cufftHandle r2c_plan;
    cufftPlan2d(&r2c_plan, num_y_real, num_x_real, CUFFT_TYPE_R2C);
    CUFFT_EXEC_R2C(r2c_plan, real_data_device_in, data_device_out);
    cufftDestroy(r2c_plan);
}


/***********************************************************************************/
/** Create block and grid size for num threads                                    **/
/***********************************************************************************/
__host__ static dim3 create_block_dim(int num) {
    dim3 block;
    block.x = MAX_NUMBER_THREADS_PER_BLOCK;
    return block;
}

__host__ static dim3 create_grid_dim(int num) {
    dim3 grid;
    // grid.x = ceil(num / MAX_N...)
    grid.x = (num + MAX_NUMBER_THREADS_PER_BLOCK - 1) / MAX_NUMBER_THREADS_PER_BLOCK;
    return grid;
}

/***********************************************************************************/
/** Get thread index in [0 : num)                                                 **/
/***********************************************************************************/
__device__ static int get_global_index() {
    return (threadIdx.x + (threadIdx.y + (threadIdx.z + (blockIdx.x + (blockIdx.y + (blockIdx.z)
            * gridDim.y) * gridDim.x) * blockDim.z) * blockDim.y) * blockDim.x);
}

// Each thread: Init values[ threadid*span..threadid*(span+1) )
__global__ static void init_random_on_GPU(int entries, CUDA_FLOAT* __restrict__ values, int span, int seed, CUDA_FLOAT_REAL max_amplitude) {
    int index = get_global_index();
    if(index < entries) {

        // Init rng
        curandState state;
        curand_init(seed, index, 0/*offset*/, &state);

        // Set random values
        int data_index_min = index * span;
        int data_index_max = data_index_min + span;

        for(int data_index = data_index_min; data_index < data_index_max; data_index++) {
            values[data_index].x = 2 * max_amplitude * (-.5f + curand_uniform(&state));
            values[data_index].y = 2 * max_amplitude * (-.5f + curand_uniform(&state));
        }
    }
}
/*__global__ static void add_random_on_GPU_real(int entries, CUDA_FLOAT_REAL* __restrict__ values, int seed, CUDA_FLOAT_REAL max_amplitude) {
    int index = get_global_index();
    if(index < entries) {

        curandState state;
        curand_init(seed, index, 0, &state);// 0 is offset
        values[index] += 2 * max_amplitude * (-.5f + curand_uniform(&state));
    }
}*/
