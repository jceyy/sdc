#include "particle_tracer.h"


/***********************************************************************************/
/** Constructor                                                                   **/
/***********************************************************************************/
particle_tracer::particle_tracer(int grid_points_x, int grid_points_y, int grid_points_z, int number_of_particles,
                                 int buffer_size_particle_position, const std::vector<CUDA_FLOAT_REAL>& cube_length,
                                 CUDA_FLOAT_REAL pDiffCoeff){

    cube_length_x = cube_length.at(0);
    cube_length_y = cube_length.at(1);
    cube_length_z = cube_length.at(2);

    iteration_index = 0;
    buffer_size = buffer_size_particle_position;
    num_particles = number_of_particles;
    particle_time = 0;
    DiffCoeff = pDiffCoeff;

    physical_x_points = grid_points_x;
    physical_y_points = grid_points_y;
    physical_z_points = grid_points_z;

    int positions_size = sizeof(CUDA_FLOAT_REAL) * num_particles * buffer_size;
    if(cudaSuccess != cudaMalloc((void**) &x_positions, positions_size)){
        EXIT_ERROR("not able to allocate particle buffer for x");
	}
    if(cudaSuccess != cudaMalloc((void**) &y_positions, positions_size)){
        EXIT_ERROR("not able to allocate particle buffer for y");
    }
    if(cudaSuccess != cudaMalloc((void**) &z_positions, positions_size)){
        EXIT_ERROR("not able to allocate particle buffer for z");
    }

    // Init random number generator for diffusion
    int rng_count = 3 * num_particles;
    if(cudaSuccess != cudaMalloc((void**) &rng_states, sizeof(curandState) * rng_count)){
        EXIT_ERROR("not able to allocate diffusion rng buffer");
    }
    setup_rng<<<create_grid_dim(rng_count), create_block_dim(rng_count)>>>(rng_states, 0, rng_count);

    //init positions at center
    init_positions_center();
}

particle_tracer::~particle_tracer(){
    cudaFree(x_positions);
    cudaFree(y_positions);
    cudaFree(z_positions);
    cudaFree(rng_states);
}


/***********************************************************************************/
/** Initialize xyz positions                                                      **/
/***********************************************************************************/
void particle_tracer::init_positions(particle_tracer::init I) {

    switch(I.type) {
    case random:
        init_positions_random();
        break;
    case center:
        init_positions_center();
        break;
    case center_sigma:
        init_positions_center_sigma(I.sigma);
        break;
    case diagonal:
        init_positions_diagonal();
        break;
    case pairs:
        init_positions_pairs();
        break;
    case coord_sigma:
        init_positions_coord_sigma(I.coord_x, I.coord_y, I.coord_z, I.sigma);
        break;
    }
}

void particle_tracer::init_positions_center()
{
    init_positions_coord_sigma(0.5*cube_length_x, 0.5*cube_length_y, 0.0, 0.0);
}

void particle_tracer::init_positions_center_sigma(CUDA_FLOAT_REAL sigma)
{
    init_positions_coord_sigma(0.5*cube_length_x, 0.5*cube_length_y, 0.0, sigma);
}

void particle_tracer::init_positions_coord_sigma(CUDA_FLOAT_REAL coord_x,
                                                 CUDA_FLOAT_REAL coord_y,
                                                 CUDA_FLOAT_REAL coord_z,
                                                 CUDA_FLOAT_REAL sigma)
{
    dim3 grid_dim = create_grid_dim(num_particles);
    dim3 block_dim = create_block_dim(num_particles);
    init_positions_coordinate_device<<<grid_dim, block_dim>>>(num_particles,
                                                              coord_x, coord_y, coord_z,
                                                              x_positions, y_positions, z_positions);
    if(sigma > 0.0) {
        dim3 grid_dim = create_grid_dim(3*num_particles);
        dim3 block_dim = create_block_dim(3*num_particles);
        diffuse_particles<<<grid_dim, block_dim>>>(num_particles, sigma,
                                                   x_positions,
                                                   y_positions,
                                                   z_positions,
                                                   rng_states);
    }
}

void particle_tracer::init_positions_diagonal()
{
    dim3 grid_dim = create_grid_dim(num_particles);
    dim3 block_dim = create_block_dim(num_particles);
    init_positions_diagonal_device<<<grid_dim, block_dim>>>(num_particles, cube_length_x, cube_length_y,
                                                     x_positions, y_positions, z_positions);
}

void particle_tracer::init_positions_pairs()
{
    if(num_particles & 0x1) EXIT_ERROR("Position pairs only possible with even number of particles.");
    dim3 grid_dim = create_grid_dim(num_particles/2);
    dim3 block_dim = create_block_dim(num_particles/2);
    init_positions_pairs_device<<<grid_dim, block_dim>>>(num_particles, cube_length_x, cube_length_y,
                                                         x_positions, y_positions, z_positions, rng_states);
}

void particle_tracer::init_positions_random()
{
    dim3 grid_dim = create_grid_dim(num_particles);
    dim3 block_dim = create_block_dim(num_particles);
    init_positions_random_device<<<grid_dim, block_dim>>>(num_particles, cube_length_x, cube_length_y,
                                                         x_positions, y_positions, z_positions, rng_states);
}

/***********************************************************************************/
/** Call timestepping for particles                                               **/
/***********************************************************************************/
void particle_tracer::step_particles(calculate_velocity_operator *vel_op, matrix_folder *f_folder, matrix_folder *g_folder, matrix_folder *F_folder, matrix_folder *G_folder, CUDA_FLOAT_REAL delta_t)
{

    if(is_buffer_full()) {
        EXIT_ERROR("buffer of particle tracer is filled! you forgot to read out the data properly!");
    }

    // new vel calc
    int num_x = 2*(f_folder->get_matrix(0)->get_matrix_size(0) - 1);
    int num_y = f_folder->get_matrix(0)->get_matrix_size(1);
    CUDA_FLOAT_REAL *linspace_z;
    cudaMalloc((void**) &linspace_z, sizeof(CUDA_FLOAT_REAL) * PARTICLE_Z_NUM);
    linspace<<<create_grid_dim(PARTICLE_Z_NUM),create_block_dim(PARTICLE_Z_NUM)>>>
            (PARTICLE_Z_NUM, linspace_z, -PARTICLE_Z_MAX, PARTICLE_Z_MAX);

    // Create memory for velocity
    int velocity_grid_size = sizeof(CUDA_FLOAT_REAL) * num_x * num_y * PARTICLE_Z_NUM;
    CUDA_FLOAT_REAL *u_x, *u_y, *u_z;
    if(        (cudaMalloc((void**) &u_x, velocity_grid_size) != cudaSuccess)
            || (cudaMalloc((void**) &u_y, velocity_grid_size) != cudaSuccess)
            || (cudaMalloc((void**) &u_z, velocity_grid_size) != cudaSuccess)) {
        EXIT_ERROR2("not able to allocate device memory for u_x:", cudaGetErrorString(cudaGetLastError()));
    }
    vel_op->calculate_operator_at(linspace_z, PARTICLE_Z_NUM, f_folder, g_folder, F_folder, G_folder, u_x, u_y, u_z);

    // free memory
    cudaFree(linspace_z);

    //...start one thread for each particle
    dim3 grid_dim = create_grid_dim(num_particles);
    dim3 block_dim = create_block_dim(num_particles);
    //...input fields
    CUDA_FLOAT_REAL* input_positions_x = x_positions + iteration_index * num_particles;
    CUDA_FLOAT_REAL* input_positions_y = y_positions + iteration_index * num_particles;
    CUDA_FLOAT_REAL* input_positions_z = z_positions + iteration_index * num_particles;
    //...output fields
    CUDA_FLOAT_REAL* output_positions_x = input_positions_x + num_particles;
    CUDA_FLOAT_REAL* output_positions_y = input_positions_y + num_particles;
    CUDA_FLOAT_REAL* output_positions_z = input_positions_z + num_particles;
    //...ask GPU to move particles


    step_particle_on_GPU<<<grid_dim,block_dim>>>(u_x, u_y, u_z,
                                                 num_x, num_y, cube_length_x,  cube_length_y,
                                                 input_positions_x, input_positions_y, input_positions_z,
                                                 delta_t, output_positions_x, output_positions_y, output_positions_z,
                                                 num_particles);
    // free memory
    cudaFree(u_x);
    cudaFree(u_y);
    cudaFree(u_z);


    if(DiffCoeff > 0.) {
        diffuse_particles<<<create_grid_dim(3*num_particles), create_block_dim(3*num_particles)>>>
                         (num_particles, sqrt(2 * DiffCoeff * delta_t),
                          output_positions_x, output_positions_y, output_positions_z,
                          rng_states);
    }

    // Restore periodic boundaries
    periodic_boundaries<<<create_grid_dim(num_particles), create_block_dim(num_particles)>>>
                       (num_particles, cube_length_x, cube_length_y,
                        output_positions_x, output_positions_y, output_positions_z);

    // Check for errors
    DBGSYNC();

    // next iteration
    iteration_index++;
    particle_time++;
}

/***********************************************************************************/
/** Calculate second moment of particle distribution                              **/
/***********************************************************************************/
CUDA_FLOAT_REAL particle_tracer::calc_secondmoment()
{

    // Create buffer memory
    CUDA_FLOAT_REAL *output_device, output_host[3];
    cudaMalloc((void**) &output_device, sizeof(CUDA_FLOAT_REAL)*3);

    // offset for particle position in xyz_positions
    int position_offset = iteration_index * num_particles;

//DBGSYNC();
    calc_moments_on_GPU<<<1, create_block_dim_power2(num_particles)>>>(
                             num_particles, x_positions + position_offset, y_positions + position_offset,
                             output_device);
//DBGSYNC();
    // Move memory to host and free device memory
    cudaMemcpy(output_host, output_device, sizeof(CUDA_FLOAT_REAL)*3, cudaMemcpyDeviceToHost);
    cudaFree(output_device);

    //cout << "2nd moment " << output_host[0] << "\t" << output_host[1] << "\t" << output_host[2] << endl;
    return output_host[2];
}

CUDA_FLOAT_REAL particle_tracer::calc_pair_secondmoment()
{
    // Create buffer memory
    CUDA_FLOAT_REAL *output_device, output_host[1];
    cudaMalloc((void**) &output_device, sizeof(CUDA_FLOAT_REAL));

    // offset for particle position in xyz_positions
    int position_offset = iteration_index * num_particles;

    calc_pair_secondmoment_on_GPU<<<1, create_block_dim_power2(num_particles)>>>
                                 (num_particles, x_positions + position_offset, y_positions + position_offset,
                                  cube_length_x, cube_length_y, output_device);

    // Move memory to host and free device memory
    cudaMemcpy(output_host, output_device, sizeof(CUDA_FLOAT_REAL), cudaMemcpyDeviceToHost);
    cudaFree(output_device);

    //cout << __FILE__ << ":" << __LINE__ << " Second moment part is: " << output_host[0] << endl;
    return output_host[0];
}

/***********************************************************************************/
/** Return if buffer is full                                                      **/
/***********************************************************************************/
bool particle_tracer::is_buffer_full() {
    return (iteration_index >= buffer_size-1);
}

/***********************************************************************************/
/** Retrieve position of particles and reset iteration.                           **/
/** Last (current) entry is not included, because it will be entry 0 after Clear  **/
/** unless iteration_index is 0.                                                  **/
/***********************************************************************************/
matrix_folder_real *particle_tracer::get_particle_positions(ClearEntries clear) {

    int num_timesteps = iteration_index;
    if(iteration_index >= buffer_size) num_timesteps = buffer_size - 1;
    else if(iteration_index < 1) num_timesteps = 1;
    int grid_size = num_particles * num_timesteps;

    std::vector<int> matrix_dimension(2);
    matrix_dimension[0] = num_particles;
    matrix_dimension[1] = num_timesteps;

    //grid_dim = create_grid_dim(grid_size);
    //block_dim = create_block_dim(grid_size);

    matrix_device_real* x = new matrix_device_real(matrix_dimension);
    matrix_device_real* y = new matrix_device_real(matrix_dimension);
    matrix_device_real* z = new matrix_device_real(matrix_dimension);
    cudaMemcpy(x->get_data(), x_positions, sizeof(CUDA_FLOAT_REAL) * grid_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(y->get_data(), y_positions, sizeof(CUDA_FLOAT_REAL) * grid_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(z->get_data(), z_positions, sizeof(CUDA_FLOAT_REAL) * grid_size, cudaMemcpyDeviceToDevice);
    /*copy_particle_positions<<<grid_dim,block_dim>>>(grid_size, x_positions, y_positions, z_positions,
                                                    x->get_data(), y->get_data(), z->get_data());*/

    if(clear == Clear && iteration_index > 0) {
        //reset particle positions and index
        dim3 grid_dim = create_grid_dim(num_particles);
        dim3 block_dim = create_block_dim(num_particles);
        reset_particle_positions<<<grid_dim,block_dim>>>(num_particles, num_timesteps, x_positions, y_positions, z_positions);
        iteration_index = 0;
    }

    matrix_folder_real* return_folder = new matrix_folder_real(3);
    return_folder->add_matrix(0, x);
	return_folder->add_matrix(1, y);	
	return_folder->add_matrix(2, z);	
	return return_folder;
}

int particle_tracer::get_particle_time() {
    return particle_time;
}

/***********************************************************************************/
/***********************************************************************************/
/** CUDA FUNCTIONS                                                                **/
/***********************************************************************************/
/***********************************************************************************/

/***********************************************************************************/
/** Create block and grid size for num threads                                    **/
/***********************************************************************************/
// Get block dim that is a power of 2, but no larger than MAX_NUMBER_THREADS_PER_BLOCK
__host__ static dim3 create_block_dim_power2(int num) {

    unsigned int dim = MAX_NUMBER_THREADS_PER_BLOCK;
    if(num < MAX_NUMBER_THREADS_PER_BLOCK) {
        // Find next power of 2
        dim = num - 1;
        dim |= dim >> 1;
        dim |= dim >> 2;
        dim |= dim >> 4;
        dim |= dim >> 8;
        dim |= dim >> 16;
        dim++;
    }

    dim3 block;
    block.x = dim;
    return block;
}

/*__host__ static dim3 create_grid_dim_power2(int num) {

    dim3 block = create_block_dim_power2(num);
    return create_grid_dim(int(block.x));// * block.y * block.z
}*/

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




/***********************************************************************************/
/** Initialize entry with zero                                                    **/
/***********************************************************************************/
/*__global__ static void init_positions_zero_device(int num_particles,
                                           CUDA_FLOAT_REAL* __restrict__ x_position,
                                           CUDA_FLOAT_REAL* __restrict__ y_position,
                                           CUDA_FLOAT_REAL* __restrict__ z_position) {
    int total_index = get_global_index();
    if(total_index < num_particles) {
        x_position[total_index] = 0.f;
        y_position[total_index] = 0.f;
        z_position[total_index] = 0.f;
	}
}*/


__global__ static void init_positions_coordinate_device(int num_particles,
                                                        CUDA_FLOAT_REAL coord_x,
                                                        CUDA_FLOAT_REAL coord_y,
                                                        CUDA_FLOAT_REAL coord_z,
                                                        CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                        CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                        CUDA_FLOAT_REAL* __restrict__ z_positions) {
    int index = get_global_index();

    if(index < num_particles) {
        x_positions[index] = coord_x;
        y_positions[index] = coord_y;
        z_positions[index] = coord_z;
    }
}


__global__ static void init_positions_diagonal_device(int num_particles, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                             CUDA_FLOAT_REAL* __restrict__ x_positions,
                                             CUDA_FLOAT_REAL* __restrict__ y_positions,
                                             CUDA_FLOAT_REAL* __restrict__ z_positions) {
    int index = get_global_index();

    if(index < num_particles) {
        x_positions[index] = (cube_length_x * index) / CUDA_FLOAT_REAL(num_particles);
        y_positions[index] = (cube_length_y * index) / CUDA_FLOAT_REAL(num_particles);
        z_positions[index] = 0.f;
    }
}


__global__ static void init_positions_pairs_device(int num_particles, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                                   CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                   CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                   CUDA_FLOAT_REAL* __restrict__ z_positions,
                                                   curandState* rng_states) {
    int index = get_global_index();

    if(index < num_particles/2) {
        CUDA_FLOAT_REAL pos_x = curand_uniform(&rng_states[index]);
        CUDA_FLOAT_REAL pos_y = curand_uniform(&rng_states[index]);
        x_positions[2*index] = x_positions[2*index+1] = cube_length_x * pos_x;
        y_positions[2*index] = y_positions[2*index+1] = cube_length_y * pos_y;
        z_positions[2*index] = z_positions[2*index+1] = 0.f;
    }
}


__global__ static void init_positions_random_device(int num_particles, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                                   CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                   CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                   CUDA_FLOAT_REAL* __restrict__ z_positions,
                                                   curandState* rng_states) {
    int index = get_global_index();

    if(index < num_particles) {
        CUDA_FLOAT_REAL pos_x = curand_uniform(&rng_states[3*index  ]);
        CUDA_FLOAT_REAL pos_y = curand_uniform(&rng_states[3*index+1]);
        CUDA_FLOAT_REAL pos_z = curand_uniform(&rng_states[3*index+2]);
        x_positions[index] = cube_length_x * pos_x;
        y_positions[index] = cube_length_y * pos_y;
        z_positions[index] = pos_z - 0.5f;
    }
}

/***********************************************************************************/
/** Copy of particle position to (complex) output array                           **/
/***********************************************************************************/
/*__global__ static void copy_particle_positions(int entries,
                                               CUDA_FLOAT_REAL* __restrict__ x_position,
                                               CUDA_FLOAT_REAL* __restrict__ y_position,
                                               CUDA_FLOAT_REAL* __restrict__ z_position,
                                               CUDA_FLOAT_REAL* __restrict__ x_output,
                                               CUDA_FLOAT_REAL* __restrict__ y_output,
                                               CUDA_FLOAT_REAL* __restrict__ z_output) {

    int total_index = get_global_index();
    if(total_index < entries) {
        x_output[total_index] = x_position[total_index];
        y_output[total_index] = y_position[total_index];
        z_output[total_index] = z_position[total_index];
	}
}*/

/***********************************************************************************/
/** Copy final positions of particles to initial positions for next round         **/
/***********************************************************************************/
__global__ static void reset_particle_positions(int num_particles, int num_timesteps,
                                                CUDA_FLOAT_REAL* __restrict__ x_position,
                                                CUDA_FLOAT_REAL* __restrict__ y_position,
                                                CUDA_FLOAT_REAL* __restrict__ z_position) {
    int total_index = get_global_index();
    int shift_index = num_timesteps * num_particles + total_index;
    if(total_index < num_particles) {
        x_position[total_index] = x_position[shift_index];
        y_position[total_index] = y_position[shift_index];
        z_position[total_index] = z_position[shift_index];
	}
}


/***********************************************************************************/
/** Linear spacing of values between min and max (num must be >1)                 **/
/***********************************************************************************/
__global__ static void linspace(int entries, CUDA_FLOAT_REAL* positions,
                                CUDA_FLOAT_REAL min, CUDA_FLOAT_REAL max) {
    int index = get_global_index();

    if(index < entries){
        positions[index] = min + index * (max - min) / (entries - 1);
    }
}

/***********************************************************************************/
/** Time stepping                                                                 **/
/** Interpolate the fluid velocity at the particle's position with trilinear      **/
/** interpolation from the known fluid velocities at the surrounding grid points. **/
/***********************************************************************************/
__device__ void vel_interpolate(CUDA_FLOAT_REAL* ux, CUDA_FLOAT_REAL* uy, CUDA_FLOAT_REAL* uz,
                                int xSize, int ySize, CUDA_FLOAT_REAL clx, CUDA_FLOAT_REAL cly,
                                CUDA_FLOAT_REAL x, CUDA_FLOAT_REAL y, CUDA_FLOAT_REAL z,
                                CUDA_FLOAT_REAL &retuX, CUDA_FLOAT_REAL &retuY, CUDA_FLOAT_REAL &retuZ) {

    int xySize = xSize * ySize;

    // x/y/z position rounding:

    // Convert from physical [0:L) to technical [0:1024) coordinates
    CUDA_FLOAT_REAL x_grid = x * xSize / clx;
    CUDA_FLOAT_REAL y_grid = y * ySize / cly;
    CUDA_FLOAT_REAL z_grid = (PARTICLE_Z_MAX + z)/(2*PARTICLE_Z_MAX) * (PARTICLE_Z_NUM - 1); // linspace goes from i=0 to i=N-1

#ifdef PERIODIC
    // Particle positions to PBC range
    x_grid -= floor(x_grid / xSize) * xSize;
    y_grid -= floor(y_grid / ySize) * ySize;
#endif

    // Calculate array index in periodic boundary conditions
    // index for left neighbor
    int iX0 = __float2int_rd(x_grid);
    int iY0 = __float2int_rd(y_grid);
    int iZ0 = __float2int_rd(z_grid);

    // Error Correction Code
#ifdef PERIODIC
    if(iX0 < 0 || iX0 >= xSize) iX0 = 0;
    if(iY0 < 0 || iY0 >= ySize) iY0 = 0;
#else
    if(iX0 < 0) iX0 = 0;
    else if(iX0 >= xSize) iX0 = xSize - 1;
    if(iY0 < 0) iY0 = 0;
    else if(iY0 >= ySize) iY0 = ySize - 1;
#endif
    if(iZ0 < 0) iZ0 = 0;
    else if(iZ0 >= PARTICLE_Z_NUM) iZ0 = PARTICLE_Z_NUM - 1;

    // index for right neighbor
#ifdef PERIODIC
    int iX1 = (iX0 + 1 >= xSize) ? (0) : (iX0 + 1);
    int iY1 = (iY0 + 1 >= ySize) ? (0) : (iY0 + 1);
#else
    int iX1 = (iX0 + 1 >= xSize) ? (iX0) : (iX0 + 1);
    int iY1 = (iY0 + 1 >= ySize) ? (iY0) : (iY0 + 1);
#endif
    int iZ1 = (iZ0 + 1 >= PARTICLE_Z_NUM) ? (iZ0) : (iZ0 + 1);


    // Indices for position evaluation
    int index0, index1;

    // interpolate in x-direction
    CUDA_FLOAT_REAL wR = x_grid - CUDA_FLOAT_REAL(iX0);
    CUDA_FLOAT_REAL wL = 1 - wR;

    index0 = iX0 + iY0 * xSize + iZ0 * xySize;
    index1 = iX1 + iY0 * xSize + iZ0 * xySize;
    CUDA_FLOAT_REAL cX00 = wL * ux[index0] + wR * ux[index1];
    CUDA_FLOAT_REAL cY00 = wL * uy[index0] + wR * uy[index1];
    CUDA_FLOAT_REAL cZ00 = wL * uz[index0] + wR * uz[index1];

    index0 = iX0 + iY1 * xSize + iZ0 * xySize;
    index1 = iX1 + iY1 * xSize + iZ0 * xySize;
    CUDA_FLOAT_REAL cX10 = wL * ux[index0] + wR * ux[index1];
    CUDA_FLOAT_REAL cY10 = wL * uy[index0] + wR * uy[index1];
    CUDA_FLOAT_REAL cZ10 = wL * uz[index0] + wR * uz[index1];

    index0 = iX0 + iY0 * xSize + iZ1 * xySize;
    index1 = iX1 + iY0 * xSize + iZ1 * xySize;
    CUDA_FLOAT_REAL cX01 = wL * ux[index0] + wR * ux[index1];
    CUDA_FLOAT_REAL cY01 = wL * uy[index0] + wR * uy[index1];
    CUDA_FLOAT_REAL cZ01 = wL * uz[index0] + wR * uz[index1];

    index0 = iX0 + iY1 * xSize + iZ1 * xySize;
    index1 = iX1 + iY1 * xSize + iZ1 * xySize;
    CUDA_FLOAT_REAL cX11 = wL * ux[index0] + wR * ux[index1];
    CUDA_FLOAT_REAL cY11 = wL * uy[index0] + wR * uy[index1];
    CUDA_FLOAT_REAL cZ11 = wL * uz[index0] + wR * uz[index1];

    // interpolate in y-direction
    wR = y_grid - CUDA_FLOAT_REAL(iY0);
    wL = 1 - wR;

    CUDA_FLOAT_REAL cX0 = wL * cX00 + wR * cX10;
    CUDA_FLOAT_REAL cY0 = wL * cY00 + wR * cY10;
    CUDA_FLOAT_REAL cZ0 = wL * cZ00 + wR * cZ10;
    CUDA_FLOAT_REAL cX1 = wL * cX01 + wR * cX11;
    CUDA_FLOAT_REAL cY1 = wL * cY01 + wR * cY11;
    CUDA_FLOAT_REAL cZ1 = wL * cZ01 + wR * cZ11;

    // interpolate in z-direction
    wR = z_grid - CUDA_FLOAT_REAL(iZ0);
    wL = 1 - wR;

    retuX = wL * cX0 + wR * cX1;
    retuY = wL * cY0 + wR * cY1;
    retuZ = wL * cZ0 + wR * cZ1;
}


__global__ void step_particle_on_GPU(CUDA_FLOAT_REAL* u_x, CUDA_FLOAT_REAL* u_y, CUDA_FLOAT_REAL* u_z,
                                     int x_grid_points, int y_grid_points,
                                     CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                     CUDA_FLOAT_REAL* input_positions_x, CUDA_FLOAT_REAL* input_positions_y,
                                     CUDA_FLOAT_REAL* input_positions_z, CUDA_FLOAT_REAL delta_t,
                                     CUDA_FLOAT_REAL* output_positions_x, CUDA_FLOAT_REAL* output_positions_y,
                                     CUDA_FLOAT_REAL* output_positions_z, int num_particles) {
    int total_index = get_global_index();

    if(total_index < num_particles) {

        //...get positions of the particle
        CUDA_FLOAT_REAL x = input_positions_x[total_index];
        CUDA_FLOAT_REAL y = input_positions_y[total_index];
        CUDA_FLOAT_REAL z = input_positions_z[total_index];

        CUDA_FLOAT_REAL vel_x, vel_y, vel_z;
#define RK4
#ifdef RK4
        // Step 1
        vel_interpolate(u_x, u_y, u_z, x_grid_points, y_grid_points, cube_length_x, cube_length_y, x, y, z, vel_x, vel_y, vel_z);

        CUDA_FLOAT_REAL dx1 = delta_t * vel_x;
        CUDA_FLOAT_REAL dy1 = delta_t * vel_y;
        CUDA_FLOAT_REAL dz1 = delta_t * vel_z;

        // Step 2
        CUDA_FLOAT_REAL xt = x + 0.5 * dx1;
        CUDA_FLOAT_REAL yt = y + 0.5 * dy1;
        CUDA_FLOAT_REAL zt = z + 0.5 * dz1;

        vel_interpolate(u_x, u_y, u_z, x_grid_points, y_grid_points, cube_length_x, cube_length_y, xt, yt, zt, vel_x, vel_y, vel_z);

        CUDA_FLOAT_REAL dx2 = delta_t * vel_x;
        CUDA_FLOAT_REAL dy2 = delta_t * vel_y;
        CUDA_FLOAT_REAL dz2 = delta_t * vel_z;

        // Step 3
        xt = x + 0.5 * dx2;
        yt = y + 0.5 * dy2;
        zt = z + 0.5 * dz2;

        vel_interpolate(u_x, u_y, u_z, x_grid_points, y_grid_points, cube_length_x, cube_length_y, xt, yt, zt, vel_x, vel_y, vel_z);

        CUDA_FLOAT_REAL dx3 = delta_t * vel_x;
        CUDA_FLOAT_REAL dy3 = delta_t * vel_y;
        CUDA_FLOAT_REAL dz3 = delta_t * vel_z;

        // Step 4
        xt = x + dx3;
        yt = y + dy3;
        zt = z + dz3;

        vel_interpolate(u_x, u_y, u_z, x_grid_points, y_grid_points, cube_length_x, cube_length_y, xt, yt, zt, vel_x, vel_y, vel_z);

        CUDA_FLOAT_REAL dx4 = delta_t * vel_x;
        CUDA_FLOAT_REAL dy4 = delta_t * vel_y;
        CUDA_FLOAT_REAL dz4 = delta_t * vel_z;

        // Sum up
        x += (dx1 + 2*(dx2+dx3) + dx4) / 6.;
        y += (dy1 + 2*(dy2+dy3) + dy4) / 6.;
        z += (dz1 + 2*(dz2+dz3) + dz4) / 6.;
#else
        vel_interpolate(u_x, u_y, u_z, x_grid_points, y_grid_points, cube_length_x, cube_length_y, x, y, z, vel_x, vel_y, vel_z);

        //use this velocity to perform euler step
        x += delta_t * vel_x;
        y += delta_t * vel_y;
        z += delta_t * vel_z;
#endif

        // set output position
        output_positions_x[total_index] = x;
        output_positions_y[total_index] = y;
        output_positions_z[total_index] = z;
    }
}

/***********************************************************************************/
/** Gaussian-distribute particles with sigma width                                **/
/***********************************************************************************/
__global__ static void diffuse_particles(int num_particles, CUDA_FLOAT_REAL sigma,
                                         CUDA_FLOAT_REAL* __restrict__ x_positions,
                                         CUDA_FLOAT_REAL* __restrict__ y_positions,
                                         CUDA_FLOAT_REAL* __restrict__ z_positions,
                                         curandState *rng_states) {
    int index = get_global_index();
    int part_index = index % num_particles;
    int dir_index = index / num_particles;

    if(dir_index < 3) {
        CUDA_FLOAT_REAL dx = sigma * curand_normal(&rng_states[index]);

        // Add noise to particle position
        switch(dir_index) {
        case 0:
            x_positions[part_index] += dx;
            break;
        case 1:
            y_positions[part_index] += dx;
            break;
        case 2:
            z_positions[part_index] += dx;
        }

    }
}

/***********************************************************************************/
/** Set periodic boundary conditions                                              **/
/***********************************************************************************/
__global__ static void periodic_boundaries(int num_particles,
                                           CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                           CUDA_FLOAT_REAL* __restrict__ x_positions,
                                           CUDA_FLOAT_REAL* __restrict__ y_positions,
                                           CUDA_FLOAT_REAL* __restrict__ z_positions) {

    int index = get_global_index();
    if(index < num_particles) {


#ifdef PERIODIC
        // Do nothing: Particles have an infinite domain,
        // only access to velocity field is finite
#else
        CUDA_FLOAT_REAL x_pos = x_positions[index];
        CUDA_FLOAT_REAL y_pos = y_positions[index];

        if(x_pos < 0) x_positions[index] = 0;
        else if(x_pos > cube_length_x) x_positions[index] = cube_length_x;
        if(y_pos < 0) y_positions[index] = 0;
        else if(y_pos > cube_length_y) y_positions[index] = cube_length_y;
#endif
        CUDA_FLOAT_REAL z_pos = z_positions[index];
        // Double negation for NaNs: get reset to top lid
        if(!(z_pos <= PARTICLE_Z_MAX)) {

            // Reflection at the top lid
            if(z_pos < 2*PARTICLE_Z_MAX) {
                z_positions[index] = 2*PARTICLE_Z_MAX - z_pos;
            } else {
                z_positions[index] = PARTICLE_Z_MAX;
            }
        } else if(z_pos < -PARTICLE_Z_MAX) {

            // Reflection at the bottom lid
            if(z_pos > -2*PARTICLE_Z_MAX) {
                z_positions[index] = -2*PARTICLE_Z_MAX - z_pos;
            } else {
                z_positions[index] = -PARTICLE_Z_MAX;
            }
        }
    }
}

/***********************************************************************************/
/** Calculate second moment                                                       **/
/** Only 512 (arch_sm 1.x)/1024 (arch_sm 2.x) particles will be tracked at a time **/
/** output[3] = { 1st moment x, y, 2nd moment }                                   **/
/***********************************************************************************/
__global__ static void calc_moments_on_GPU(int num_particles,
                                                 CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                 CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                 CUDA_FLOAT_REAL* output) {
    __shared__ CUDA_FLOAT_REAL buffer_x[MAX_NUMBER_THREADS_PER_BLOCK];
    __shared__ CUDA_FLOAT_REAL buffer_y[MAX_NUMBER_THREADS_PER_BLOCK];
    __shared__ CUDA_FLOAT_REAL buffer_x2[MAX_NUMBER_THREADS_PER_BLOCK];


    // All threads must be active for syncthreads()
    const int index = get_global_index();
    const int blockSize = blockDim.x * blockDim.y * blockDim.z;//must be power of 2

    // STEP 1: Calculate <x>, <y> and <x^2+y^2> for one thread
    CUDA_FLOAT_REAL xavg = 0;
    CUDA_FLOAT_REAL yavg = 0;
    CUDA_FLOAT_REAL x2avg = 0;
    for(int part_index = index; part_index < num_particles; part_index += blockSize) {
        const CUDA_FLOAT_REAL xpos = x_positions[part_index];
        const CUDA_FLOAT_REAL ypos = y_positions[part_index];
        xavg += xpos;
        yavg += ypos;
        x2avg += xpos*xpos + ypos*ypos;
    }
    buffer_x[index] = xavg / num_particles;
    buffer_y[index] = yavg / num_particles;
    buffer_x2[index] = x2avg / num_particles;

    // STEP 2: Cross-thread sum to get <x>, <y> and <x^2+y^2> in buffer[0]
    int mask = 0;
    for(int i = 1; i < blockSize; i <<= 1) {
        __syncthreads();
        mask |= i;

        if(((index & mask) == 0) && (index + i < num_particles)) {
            buffer_x[index] += buffer_x[index + i];
            buffer_y[index] += buffer_y[index + i];
            buffer_x2[index] += buffer_x2[index + i];
        }
    }
    __syncthreads();

    // STEP 3: Store output in array (done by 1st thread)
    if(index == 0) {
        CUDA_FLOAT_REAL xavg = buffer_x[0];
        CUDA_FLOAT_REAL yavg = buffer_y[0];
        output[0] = xavg;
        output[1] = yavg;
        output[2] = buffer_x2[0] - xavg*xavg - yavg*yavg;
    }
}

/***********************************************************************************/
/** Calculate average second moment of 2 particles                                **/
/** Only one thread block (MAX_NUMBER_THREADS_PER_BLOCK (512) threads) is needed. **/
/** output[0] = < |x_(2i)-x_(2i+1)|^2 >                                           **/
/***********************************************************************************/
__global__ static void calc_pair_secondmoment_on_GPU(int num_particles,
                                                     CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                     CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                     CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                                     CUDA_FLOAT_REAL* output) {
    __shared__ CUDA_FLOAT_REAL buffer[MAX_NUMBER_THREADS_PER_BLOCK];

    const int index = get_global_index();
    const int blockSize = blockDim.x * blockDim.y * blockDim.z;

    // Fill buffer with x^2+y^2-values
    if(index < blockSize) {
        buffer[index] = 0;

        for(int part_index = 2*index; part_index < num_particles-1; part_index += 2*blockSize) {
            CUDA_FLOAT_REAL dx = fabs(x_positions[part_index+1] - x_positions[part_index]);
            CUDA_FLOAT_REAL dy = fabs(y_positions[part_index+1] - y_positions[part_index]);
    #ifdef PERIODIC
            if(dx > 0.5*cube_length_x) dx = cube_length_x - dx;
            if(dy > 0.5*cube_length_y) dy = cube_length_y - dy;
    #endif
            buffer[index] += dx * dx + dy * dy;
        }
    }

    // Add up buffer at buffer[0]
    // Cross-thread average
    int mask = 0;
    for(int i = 1; i < blockSize; i <<= 1) {
        __syncthreads();
        mask |= i;  // 0001, 0011, 0111, ...

        if(((index & mask) == 0) && (index + i < blockSize)) {
            buffer[index] += buffer[index + i];
        }
    }
    __syncthreads();

    // 1st thread stores output in array
    if(index == 0) {
        output[0] = buffer[0] / (num_particles / 2);
    }
}

__global__ void setup_rng(curandState *rng_states, int seed, int num_rngs)
{
    int index = get_global_index();

    if(index < num_rngs) {
        curand_init(seed, index, 0/*offset*/, &rng_states[index]);
    }
}

// Snippets
/*if(iX0 < 0) iX0 = (xSize + (iX0 % xSize)) % xSize;
else if(iX0 >= xSize) iX0 = iX0 % xSize;
if(iY0 < 0) iY0 = (ySize + (iY0 % ySize)) % ySize;
else if(iY0 >= ySize) iY0 = iY0 % ySize;
if(iZ0 < 0) iZ0 = 0;
else if(iZ0 >= PARTICLE_Z_NUM - 1) iZ0 = PARTICLE_Z_NUM - 2;*/


/*if(x_pos < 0 || x_pos >= cube_length_x) {
    x_positions[index] = x_pos - cube_length_x * floor(x_pos / cube_length_x);
}
if(y_pos < 0 || y_pos >= cube_length_y) {
    y_positions[index] = y_pos - cube_length_y * floor(y_pos / cube_length_y);
}*/
