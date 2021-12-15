#ifndef PARTICLE_TRACER_H
#define PARTICLE_TRACER_H

/***********************************************************************************/
/** Particle tracer                                                               **/
/** Traces N particles in the velocity field. Position is measured in (M,N,L)-    **/
/** space. Timestepping via Runge-Kutta 4th order.                                **/
/** When is_buffer_full(), get_particle_positions(particle_tracer::Clear)         **/
/** is needed to empty the buffer.                                                **/
/***********************************************************************************/

#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include "../cuda_defines.h"
#include "../matrix/matrix_device.h"
#include "../matrix/matrix_folder.h"
#include "../operator/calculate_velocity_operator.h"


class particle_tracer {

private:
	int iteration_index;
	int buffer_size;
	int num_particles;
    int particle_time;

	int physical_x_points;
	int physical_y_points;
	int physical_z_points;

    // Diffusion coefficient
    CUDA_FLOAT_REAL DiffCoeff;
    curandState *rng_states;

protected:
    // Maybe merge position-arrays in one big?
    CUDA_FLOAT_REAL* x_positions;
	CUDA_FLOAT_REAL* y_positions;
    CUDA_FLOAT_REAL* z_positions;

	CUDA_FLOAT_REAL cube_length_x;
	CUDA_FLOAT_REAL cube_length_y;
	CUDA_FLOAT_REAL cube_length_z;

public:

    enum ClearEntries { noClear, Clear };
    enum initType { random, center, center_sigma, diagonal, pairs, coord_sigma};

    // Init-struct stores all information for initializing a particle tracer
    struct init {
        int num_particles;
        initType type;
        CUDA_FLOAT_REAL coord_x, coord_y, coord_z;  // only for coord_sigma
        CUDA_FLOAT_REAL sigma;  // only for center_sigma and coord_sigma
    };

    particle_tracer(int grid_points_x, int grid_points_y, int grid_points_z, int number_of_particles,
                    int buffer_size_particle_position, const std::vector<CUDA_FLOAT_REAL>& cube_length, CUDA_FLOAT_REAL pDiffCoeff);
	~particle_tracer();

    //step particles through space
    void step_particles(calculate_velocity_operator* vel_op_init, matrix_folder* f_folder, matrix_folder* g_folder, matrix_folder* F_folder, matrix_folder* G_folder,
                        CUDA_FLOAT_REAL delta_t);
    CUDA_FLOAT_REAL calc_secondmoment();
    CUDA_FLOAT_REAL calc_pair_secondmoment();
	
    //init particle positions
    void init_positions(particle_tracer::init I);
    void init_positions_center();
    void init_positions_center_sigma(CUDA_FLOAT_REAL sigma);
    void init_positions_coord_sigma(CUDA_FLOAT_REAL coord_x, CUDA_FLOAT_REAL coord_y, CUDA_FLOAT_REAL coord_z, CUDA_FLOAT_REAL sigma);
    void init_positions_diagonal();
    void init_positions_pairs();
    void init_positions_random();

    bool is_buffer_full();

	/**
	* returns a matrix_folder with particle's positions
	* folder has three matrix_devices (one for x, y and z)
	* columns are the different particles
	* rows are the different timepoints
	*/
    matrix_folder_real* get_particle_positions(ClearEntries clear);
    int get_particle_time();
};


__host__ static dim3 create_block_dim_power2(int num);
__host__ static dim3 create_grid_dim_power2(int num);
__host__ static dim3 create_block_dim(int num);
__host__ static dim3 create_grid_dim(int num);
__device__ static int get_global_index();

__global__ static void init_positions_coordinate_device(int num_particles,
                                                        CUDA_FLOAT_REAL coord_x,
                                                        CUDA_FLOAT_REAL coord_y,
                                                        CUDA_FLOAT_REAL coord_z,
                                                        CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                        CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                        CUDA_FLOAT_REAL* __restrict__ z_positions);
__global__ static void init_positions_diagonal_device(int num_particles, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                             CUDA_FLOAT_REAL* __restrict__ x_positions,
                                             CUDA_FLOAT_REAL* __restrict__ y_positions,
                                             CUDA_FLOAT_REAL* __restrict__ z_positions);
__global__ static void init_positions_pairs_device(int num_particles, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                                   CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                   CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                   CUDA_FLOAT_REAL* __restrict__ z_positions,
                                                   curandState* rng_states);
__global__ static void init_positions_random_device(int num_particles, CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                                   CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                   CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                   CUDA_FLOAT_REAL* __restrict__ z_positions,
                                                   curandState* rng_states);

/*__global__ static void copy_particle_positions(int entries,
                                               CUDA_FLOAT_REAL* __restrict__ x_position,
                                               CUDA_FLOAT_REAL* __restrict__ y_position,
                                               CUDA_FLOAT_REAL* __restrict__ z_position,
                                               CUDA_FLOAT* __restrict__ x_output,
                                               CUDA_FLOAT* __restrict__ y_output,
                                               CUDA_FLOAT* __restrict__ z_output);*/

__global__ static void reset_particle_positions(int num_particles, int num_timesteps,
                                                CUDA_FLOAT_REAL* __restrict__ x_position,
                                                CUDA_FLOAT_REAL* __restrict__ y_position,
                                                CUDA_FLOAT_REAL* __restrict__ z_position);

//ifdef PARTICLE_INTERPOLATE_Z
__global__ static void linspace(int entries, CUDA_FLOAT_REAL* positions,
                                CUDA_FLOAT_REAL min, CUDA_FLOAT_REAL max);
//endif

__device__ void vel_interpolate(CUDA_FLOAT_REAL* ux, CUDA_FLOAT_REAL* uy, CUDA_FLOAT_REAL* uz,
                                int xSize, int ySize, CUDA_FLOAT_REAL clx, CUDA_FLOAT_REAL cly,
                                CUDA_FLOAT_REAL x, CUDA_FLOAT_REAL y, CUDA_FLOAT_REAL z,
                                CUDA_FLOAT_REAL &retuX, CUDA_FLOAT_REAL &retuY, CUDA_FLOAT_REAL &retuZ);

__global__ void step_particle_on_GPU(CUDA_FLOAT_REAL* u_x, CUDA_FLOAT_REAL* u_y, CUDA_FLOAT_REAL* u_z,
                                     int x_grid_points, int y_grid_points,
                                     CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                     CUDA_FLOAT_REAL* input_positions_x, CUDA_FLOAT_REAL* input_positions_y,
                                     CUDA_FLOAT_REAL* input_positions_z, CUDA_FLOAT_REAL delta_t,
                                     CUDA_FLOAT_REAL* output_positions_x, CUDA_FLOAT_REAL* output_positions_y,
                                     CUDA_FLOAT_REAL* output_positions_z, int num_particles);

__global__ static void diffuse_particles(int num_particles, CUDA_FLOAT_REAL sigma,
                                         CUDA_FLOAT_REAL* __restrict__ x_positions,
                                         CUDA_FLOAT_REAL* __restrict__ y_positions,
                                         CUDA_FLOAT_REAL* __restrict__ z_positions,
                                         curandState *rng_states);

__global__ static void periodic_boundaries(int num_particles,
                                           CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                           CUDA_FLOAT_REAL* __restrict__ x_positions,
                                           CUDA_FLOAT_REAL* __restrict__ y_positions,
                                           CUDA_FLOAT_REAL* __restrict__ z_positions);

__global__ static void calc_moments_on_GPU(int num_particles,
                                           CUDA_FLOAT_REAL* __restrict__ x_positions,
                                           CUDA_FLOAT_REAL* __restrict__ y_positions,
                                           CUDA_FLOAT_REAL* output);

__global__ static void calc_pair_secondmoment_on_GPU(int num_particles,
                                                     CUDA_FLOAT_REAL* __restrict__ x_positions,
                                                     CUDA_FLOAT_REAL* __restrict__ y_positions,
                                                     CUDA_FLOAT_REAL cube_length_x, CUDA_FLOAT_REAL cube_length_y,
                                                     CUDA_FLOAT_REAL* output);

__global__ static void setup_rng(curandState *rng_states, int seed, int num_rngs);
#endif


