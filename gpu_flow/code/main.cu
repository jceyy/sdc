//system includes
#include <iostream>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//cuda includes
#include <cuda.h>
#include <cufft.h>
//include <curand.h>

//my includes
#include "cuda_defines.h"
#include "util/init.h"
#include "util/util.h"
#include "util/coeff.h"
#include "util/inputparser.h"
#include "matrix/matrix_host.h"
#include "matrix/matrix_device.h"
#include "matrix/matrix_folder.h"
#include "output/matrix_folder_writer.h"
#ifdef USE_SERVER
#include "output/server_writer.h"
#endif
#include "output/datapoint_writer.h"
#include "output/track_writer.h"
#include "operator/calculate_velocity_operator.h"
#include "operator/calculate_temperature_operator.h"
#include "operator/linear_implicit_operator.h"
#include "operator/inverse_B_matrix.h"
#include "timestepping/implicit_explicit_stepping.h"
#ifdef USE_PARTICLES
#include "particle/particle_tracer.h"
#endif
#ifdef CALC_STRUCTFN
#include "postProcessing/structure_function_2d.h"
#endif
#ifdef CALC_ENERGYFN
#include "postProcessing/energy_spectrum_2d.h"
#endif
#include "operator/freeze_operator.h"
#include "operator/masktemp.h"
#include "operator/masking.h" // Absent m



int main(int argc, char** argv) {

    char tmp_filename[FILENAME_LEN];
    char tmp_string[FILENAME_LEN];
    char tmp_rundescr[256];

    /***********************************************************************************/
    /** Read command line arguments                                                   **/
    /***********************************************************************************/
    inputparser ip(argc, argv);

    //...the cuda device
    int cuda_device = 0;
    ip.get<int>(cuda_device, "-device", inputparser::optional);
 
    // read target path
    char dirbuffer[FILENAME_LEN]; strcpy(dirbuffer, "");
    char *dirbuffer_list[1] = { dirbuffer };
    ip.getstrings(dirbuffer_list, FILENAME_LEN, 1, "-dir", inputparser::optional);

    // ...number of modes in x and y direction
    vector<int> dim_real;
    ip.get<int>(dim_real, 3, "-nmodes", inputparser::mandatory);
    int M = dim_real.at(0);    // number of modes in x direction => columns (first index)
    int N = dim_real.at(1);    // number of modes in y direction => rows (second index)
    int L = dim_real.at(2);    // number of modes in z direction => the number of vertical chandrasekhar/sinus functions (third index)

    // ...cube length in x and y direction
    // Length of the cube in physical space as multiples of the vertical length, which is equal to one
    vector<CUDA_FLOAT_REAL> cube_length;
    ip.get<CUDA_FLOAT_REAL>(cube_length, 2, "-clength", inputparser::mandatory);
    double cube_length_x = cube_length.at(0);
    double cube_length_y = cube_length.at(1);
    //...z length is fixed to 1 because of special ansatz functions
    cube_length.resize(3); cube_length.at(2) = 1.0;

    // ...initialization for theta
    matrix_device_init::initType       theta_init = matrix_device_init::random;
    if(ip.getopt("-init_rectangle"))   theta_init = matrix_device_init::rectangle;
    if(ip.getopt("-init_dislocation")) theta_init = matrix_device_init::dislocation;
    if(ip.getopt("-init_isr"))         theta_init = matrix_device_init::isr;
    if(ip.getopt("-init_sv"))          theta_init = matrix_device_init::sv;
    if(ip.getopt("-init_cr"))          theta_init = matrix_device_init::cr;
    if(ip.getopt("-init_giantspiral")) theta_init = matrix_device_init::giantspiral;
    if(ip.getopt("-init_test"))        theta_init = matrix_device_init::test;
    double init_mode = cube_length_x/2.0;
    ip.get<double>(init_mode, "-init_mode", inputparser::optional);

    //...the rayleigh number
    CUDA_FLOAT_REAL rayleigh_number = 1900;//0.72*1708.0+1708.0; // 1708.0 is critical
    ip.get<CUDA_FLOAT_REAL>(rayleigh_number, "-ra", inputparser::mandatory);
	
    //...the prandtl number
    CUDA_FLOAT_REAL prandtl_number = 1.0;//0.96;
    ip.get<CUDA_FLOAT_REAL>(prandtl_number, "-pr", inputparser::mandatory);

    //...the timestepping
    CUDA_FLOAT_REAL delta_t = 0.001;
    CUDA_FLOAT_REAL final_t = delta_t;
    ip.get<CUDA_FLOAT_REAL>(delta_t, "-dt", inputparser::mandatory);
    ip.get<CUDA_FLOAT_REAL>(final_t, "-t", inputparser::mandatory);

    //...the number of files (for storage of intermediate results, in case of system crashes etc.)
    int number_of_files = 1;
    ip.get<int>(number_of_files, "-nfiles", inputparser::optional);

    //...penalization parameter
    CUDA_FLOAT_REAL eta = 0.01;
    ip.get<CUDA_FLOAT_REAL>(eta, "-eta", inputparser::optional);
#ifndef PENALIZED
    eta = 0;
#endif

    //...input datafiles
    char infilename_theta[FILENAME_LEN]; strcpy(infilename_theta, "");
    char infilename_f[FILENAME_LEN];     strcpy(infilename_f, "");
    char infilename_g[FILENAME_LEN];     strcpy(infilename_g, "");
    char infilename_F[FILENAME_LEN];     strcpy(infilename_F, "");
    char infilename_G[FILENAME_LEN];     strcpy(infilename_G, "");
    char *infiles[5] = { infilename_theta, infilename_f, infilename_g, infilename_F, infilename_G };
    ip.getstrings(infiles, FILENAME_LEN, 5, "-files", inputparser::optional);

    //...rng seed
    int seed = 0;
    ip.get<int>(seed, "-seed", inputparser::optional);

    //...mean flow forcing
    CUDA_FLOAT_REAL mean_F = 0.0;
    ip.get<CUDA_FLOAT_REAL>(mean_F, "-mean_F", inputparser::optional);

#ifdef USE_SERVER
    //...server port
    int udp_port = 1234 + cuda_device;

    ip.get<int>(udp_port, "-port", inputparser::optional);
#endif

    // Particles

#ifdef USE_PARTICLES
    //...particle count
    particle_tracer::init part_init;
    part_init.num_particles = 0;
    ip.get<int>(part_init.num_particles, "-npart", inputparser::optional);
    part_init.type = particle_tracer::random;
    if(ip.getopt("-part_center"))       part_init.type = particle_tracer::center;
    if(ip.getopt("-part_center_sigma")) {
        part_init.type = particle_tracer::center_sigma;
        part_init.sigma = 0.125;
    }
    if(ip.getopt("-part_diagonal"))     part_init.type = particle_tracer::diagonal;
    if(ip.getopt("-part_pairs"))        part_init.type = particle_tracer::pairs;
    if(ip.getopt("-part_coord_sigma")) {
        part_init.type = particle_tracer::coord_sigma;
        vector<CUDA_FLOAT_REAL> coord(2);
        ip.get<CUDA_FLOAT_REAL>(coord, 2, "-part_coord_sigma", inputparser::mandatory);
        part_init.coord_x = coord.at(0);
        part_init.coord_y = coord.at(1);
        part_init.coord_z = 0.0;
        part_init.sigma = 0.125;
    }
    if(ip.getopt("-part_sigma")) {
        ip.get<CUDA_FLOAT_REAL>(part_init.sigma, "-part_sigma", inputparser::mandatory);
    }

    // Enable writing tracks
    bool write_tracks = ip.getopt("-write_tracks_every");

    // Set track write period (in simulation steps)
    int track_write_period = 1;
    ip.get<int>(track_write_period, "-write_tracks_every", inputparser::optional);

    // Calculate second moment of distribution
    bool calc_secondmoment = ip.getopt("-calc_secondmoment");
#ifdef CALC_SECONDMOMENT // compatibility with old scripts
    calc_secondmoment = true;
#endif // CALC_SECONDMOMENT



    //...the lewis number
    int le_count = 1;
    ip.get<int>(le_count, "-le_count", inputparser::optional);

    vector<CUDA_FLOAT_REAL> lewis_numbers(0);
    if(ip.getopt("-le")) {
        ip.get<CUDA_FLOAT_REAL>(lewis_numbers, le_count, "-le", inputparser::optional);
        // lewis count is outdated at this point
    }

    // Maintain compatibility:
#ifdef MULTITRACER
    if(lewis_numbers.size() < 1) {
        //                0    1    2    3     4     5     6     7     8     9
        double tmp[10] = {1e1, 3e0, 1e0, 3e-1, 1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-5};
        for(int i = 0; i < 10; ++i) lewis_numbers.push_back(tmp[i]);
    }
#endif
#endif

    //...frequency for temperature snapshots
    int snapshot_freq = 0;
    ip.get<int>(snapshot_freq, "-snapshot_freq", inputparser::optional);

    // Possibility to stop timestepping
    bool do_timestep = !ip.getopt("-no_step_time");

    // Timestep at which outside of circle with radius 0.8
    // will be frozen

    int freeze_circle =0;
    ip.get<int>(freeze_circle, "-freeze_circle", inputparser::optional);

// Y changes
//Activating the heating mask

    int temp_circle = 0;
    ip.get<int>(temp_circle, "-temp_circle", inputparser::optional); //unnecessary for the mom, change stuff in makefile and inputparser

    /***********************************************************************************/
    /** Create working directory                                                      **/
    /***********************************************************************************/
    // if full file path is supplied, use that; else set up new folder in data directory
    {
        char workingdir[FILENAME_LEN];
        if(dirbuffer[0] == 0) {
            strcpy(workingdir, "../data");
        } else if(dirbuffer[0] == '/' || dirbuffer[0] == '~') {
            strncpy(workingdir, dirbuffer, FILENAME_LEN - 1);
            workingdir[FILENAME_LEN - 1] = 0;
        } else {
            strcpy(workingdir, "../data/");
            strncat(workingdir, dirbuffer, FILENAME_LEN - 9);
            workingdir[FILENAME_LEN - 1] = 0;
        }

    mkdir(workingdir, 0777);     
	if(chdir(workingdir) != 0) {
            EXIT_ERROR("ERROR: not able to create and change to output directory!");
        }
    }

    /***********************************************************************************/
    /** Set GPU device                                                                **/
    /***********************************************************************************/
    int cuda_device_max = 1;
    cudaGetDeviceCount(&cuda_device_max);
    cout << "number of GPUs: " << cuda_device_max << endl;
    if(cuda_device >= cuda_device_max || cuda_device < 0){
        EXIT_ERROR("ERROR: please select a valid CUDA device!");
    }
    cudaSetDevice(cuda_device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    DBGSYNC();
    cout << "device name: " << deviceProp.name << endl;


    /***********************************************************************************/
    /** Create data fields                                                            **/
    /***********************************************************************************/
    //number of modes in Fourier space
    cout << "number of spectral coefficients: " << M/2+1 << " x " << N << " x " << L << endl;
    vector<int> dimensions(3);
    dimensions[0] = M/2+1; dimensions[1] = N; dimensions[2] = L;

    //number of iterations
    int number_of_iterations = iCeil(final_t / delta_t);
    int write_file = (number_of_files > 0) ? (number_of_iterations / number_of_files) : 0;
    cout << "number of iterations: " << number_of_iterations << " with dt " << delta_t << endl;
    cout << "writing file each " << write_file << "th iteration" << endl;

    matrix_folder* theta_folder;
    matrix_folder* f_folder;
    matrix_folder* g_folder;
    matrix_folder* F_folder;
    matrix_folder* G_folder;

    // Read files if all filenames are present
    if(infilename_theta[0] == 0 || infilename_f[0] == 0 || infilename_g[0] == 0
                                || infilename_F[0] == 0 || infilename_G[0] == 0) {

        //create data in device memory
        vector<int> one_dim(3);
        one_dim[0]=1; one_dim[1]=1; one_dim[2]=L;

        //...init temperature field with small random data
        matrix_device* theta_device = new matrix_device(dimensions);
        matrix_device_init::init_random(theta_device, seed);
        theta_folder = new matrix_folder(1);
        theta_folder->add_matrix(0, theta_device);

        matrix_device* f_device = new matrix_device(dimensions);
        matrix_device_init::init_zeros(f_device);
        f_folder = new matrix_folder(1);
        f_folder->add_matrix(0, f_device);

        matrix_device* g_device = new matrix_device(dimensions);
        matrix_device_init::init_zeros(g_device);
        g_folder = new matrix_folder(1);
        g_folder->add_matrix(0, g_device);

        matrix_device* F_device = new matrix_device(one_dim);
        matrix_device_init::init_zeros(F_device);
        F_folder = new matrix_folder(1);
        F_folder->add_matrix(0, F_device);

        matrix_device* G_device = new matrix_device(one_dim);
        matrix_device_init::init_zeros(G_device);
        G_folder = new matrix_folder(1);
        G_folder->add_matrix(0, G_device);

        // Set initial values for theta
        switch(theta_init) {
        case matrix_device_init::random:
            // random init is done at creation (for all levels!)
            break;
        case matrix_device_init::rectangle:
            matrix_device_init::init_rectangle(theta_device, (cube_length_x/(M-1.0))*(M-5.0)/2.0, 0);
            break;
        case matrix_device_init::dislocation:
            matrix_device_init::init_dislocation(theta_device, init_mode, 0);
            break;
        case matrix_device_init::isr:
            matrix_device_init::init_isr(theta_device, init_mode, 0);
            break;
        case matrix_device_init::sv:
            matrix_device_init::init_sv(theta_device, init_mode, 0.2);
            break;
        case matrix_device_init::cr:
            matrix_device_init::init_cr(theta_device, init_mode, 0.8);
            break;
        case matrix_device_init::giantspiral:
            matrix_device_init::init_giantspiral(theta_device, init_mode);
            break;
        case matrix_device_init::zeros:
            matrix_device_init::init_zeros(theta_device);
            break;
        case matrix_device_init::test:
            matrix_device_init::init_test(f_device, g_device, theta_device, F_device, G_device);
            break;
        }

    } else { // Read binary input files

        cout << "...reading init data from binary file" << endl;

        theta_folder = matrix_folder_writer::read_binary_file(infilename_theta);
        if(!matrix_has_size(theta_folder, dimensions[0], dimensions[1], dimensions[2])) {
            EXIT_ERROR("matrix theta_folder does not have the right dimension");
        }

        f_folder = matrix_folder_writer::read_binary_file(infilename_f);
        if(!matrix_has_size(f_folder, dimensions[0], dimensions[1], dimensions[2])) {
            EXIT_ERROR("matrix f_folder does not have the right dimension");
        }

        g_folder = matrix_folder_writer::read_binary_file(infilename_g);
        if(!matrix_has_size(g_folder, dimensions[0], dimensions[1], dimensions[2])) {
            EXIT_ERROR("matrix g_folder does not have the right dimension");
        }

        F_folder = matrix_folder_writer::read_binary_file(infilename_F);
        if(!matrix_has_size(F_folder, 1, 1, L)) {
            EXIT_ERROR("matrix F_folder does not have the right dimension");
        }

        G_folder = matrix_folder_writer::read_binary_file(infilename_G);
        if(!matrix_has_size(G_folder, 1, 1, L)) {
            EXIT_ERROR("matrix G_folder does not have the right dimension");
        }

        cout << "...reading init data from binary file finished" << endl;
    }

    // Create iteration vector
    matrix_folder* iteration_vector[5];
    iteration_vector[0] = theta_folder;
    iteration_vector[1] = f_folder;
    iteration_vector[2] = g_folder;
    iteration_vector[3] = F_folder;
    iteration_vector[4] = G_folder;

    // Force the signal to be real
    make_real_signal(iteration_vector);


    /***********************************************************************************/
    /** Write simulation parameters                                                   **/
    /***********************************************************************************/
    {
        ofstream log("simulation.log");
        for(int i = 0; i < 2; i++) {
            ostream& s = (i==0)?log:cout;
            s << "GPU device " << cuda_device << ": " << deviceProp.name << endl;
            s << "Simulation of Rayleigh-Benard convection" << endl;
            s << "Rayleigh-Number: " << rayleigh_number << endl;
            s << "Prandtl-Number: " << prandtl_number << endl;
            s << "delta t: " << delta_t << endl;
            s << "final t: " << final_t << endl;
            s << "number of files: " << number_of_files << endl;
            s << "number of iterations: " << number_of_iterations << endl;
            s << "eta(penalization): " << eta << endl;
            s << "cube length x direction: " << cube_length_x << endl;
            s << "cube length y direction: " << cube_length_y << endl;
            s << "cube length z direction: " << 1.0 << endl;
            s << "number of x modes: " << M << endl;
            s << "number of y modes: " << N << endl;
            s << "number of z modes: " << L << endl;
            s << "command string: ";
            for(int argi = 0; argi < argc; argi++) s << argv[argi] << ' ';
            s << endl;
        }
        log.close();
    }

    /***********************************************************************************/
    /** Create Chandrasekhar coefficients                                             **/
    /***********************************************************************************/
    Coeff<double> coeff(dimensions.at(2));

    /***********************************************************************************/
    /** Initialize operators                                                          **/
    /***********************************************************************************/
    //DBGOUT("Initialize operators");

    //...inital velocity field
    calculate_velocity_operator* vel_op_init = calculate_velocity_operator::init(dimensions, cube_length);
    //...inital temperature field
    calculate_temperature_operator* temp_op_init = calculate_temperature_operator::init(dimensions, cube_length);

    /***********************************************************************************/
    /** Create operators for loop                                                     **/
    /***********************************************************************************/
    //init implicit splitting operator on a periodic domain
    linear_implicit_operator* lin_split_implicit =
    linear_implicit_operator::init_operator(dimensions, cube_length, rayleigh_number, prandtl_number, coeff, delta_t);
    inverse_B_matrix* inverseB =
    inverse_B_matrix::init_operator(dimensions, cube_length, prandtl_number, coeff);
    nonlinear_operator_rayleigh_noslip* nonlin_op =
    nonlinear_operator_rayleigh_noslip::init_operator(dimensions, cube_length, prandtl_number, coeff, eta);
    implicit_explicit_step* stepper =
    implicit_explicit_step::init_timestepping(lin_split_implicit, inverseB, nonlin_op, delta_t);

    freeze_operator* freeze_op = NULL;
    if(freeze_circle) {
        freeze_op = freeze_operator::init_operator(dimensions);
    }

	//m changes
    masking* interactionMask = NULL;
    interactionMask = masking::init_the_mask(dimensions);
    typedef net_dataSend NetUserInput; NetUserInput netUserInput;

	//Y changes
	masktemp* addtemp_op = NULL;
    if(temp_circle) {
        addtemp_op = masktemp::init_operator(dimensions);
    }

    #ifdef CALC_STRUCTFN
        structure_function_2d* structure_op = new structure_function_2d(dimensions, cube_length);
    #endif
    #ifdef CALC_ENERGYFN
        energy_spectrum_2d* energy_op = new energy_spectrum_2d(dimensions, cube_length);
    #endif

    #ifdef TIMER
        // Timer for performance measurement
        Timer stepTimer;
        Timer partTimer;
        Timer outputTimer;
    #endif

    /***********************************************************************************/
    /** Create particle tracer                                                        **/
    /***********************************************************************************/
    #ifdef USE_PARTICLES
        //DBGOUT("Start particle tracer");
        const int buffer_size_particle_position = 256;

        // Create one particle tracer for each Lewis number
        vector<particle_tracer*> tracers(lewis_numbers.size());
        if(part_init.num_particles > 0) {
            for(size_t i = 0; i < tracers.size(); ++i) {
                tracers.at(i) = new particle_tracer(M, N, L, part_init.num_particles, buffer_size_particle_position, cube_length, lewis_numbers.at(i));
                tracers.at(i)->init_positions(part_init);
            }
        } else {
            tracers.resize(0);
        }
        const int tracer_count = tracers.size();

        // Create multiple track writers
        vector<track_writer*> trackwriters;
        if(write_tracks) {
            for(int i = 0; i < tracer_count; ++i) {
                sprintf(tmp_rundescr, "Particle tracks.  Simulation: Ra=%.1lf Pr=%.1lf Size=%.1lfx%.1lfx1.0  Technical: Modes=%dx%dx%d dt=%.3lf  Tracks: Le=%.0le ",
                        rayleigh_number, prandtl_number, cube_length_x, cube_length_y,
                        M, N, L, delta_t, lewis_numbers.at(i));
                trackwriters.push_back(new track_writer(tmp_rundescr, track_write_period));
            }
        }

        // Create multiple secondmoment writers
        vector<datapoint_writer*> second_moment_writers;
        if(calc_secondmoment) {
            for(int i = 0; i < tracer_count; ++i) {
                sprintf(tmp_filename, "secondmoment_%d.dat", i);
                sprintf(tmp_string, "Le %.9f", lewis_numbers.at(i));
                second_moment_writers.push_back(new datapoint_writer(tmp_filename, FILENAME_LEN, tmp_string));
            }
        }
    #endif  // USE_PARTICLES

    /***********************************************************************************/
    /** Open server                                                                   **/
    /***********************************************************************************/
    #ifdef USE_SERVER
        //DBGOUT("Start server");
        server_writer serverWriter(udp_port);
        serverWriter.setDeviceName(deviceProp.name);
        serverWriter.setCL(cube_length_x, cube_length_y, 1.0);
        serverWriter.setNM(M, N, L);
        serverWriter.setRaPrEta(rayleigh_number, prandtl_number, eta);
        serverWriter.setdttFinal(delta_t, final_t);
        serverWriter.setTemperature(theta_folder, temp_op_init);

        #ifdef USE_PARTICLES
            if(tracer_count) serverWriter.setParticles(tracers.at(tracer_count / 2), cube_length_x, cube_length_y);
        #endif
    #endif


    // Check for errors in init phase
    DBGSYNC();
    cout << "Start time loop" << endl;

    /***********************************************************************************/
    /** ***************************************************************************** **/
    /** Time loop                                                                     **/
    /** ***************************************************************************** **/
    /***********************************************************************************/
    
    for(int i = 1; i <= number_of_iterations; i++) {




        /***********************************************************************************/
        /** Time step                                                                     **/
        /***********************************************************************************/
        if(do_timestep) {
        
        	// Special: Force mean flow
            if(mean_F) {
                force_mean(iteration_vector[3], mean_F);
            }
            
        	if(freeze_circle)
			{
				freeze_op->calculate_operator(iteration_vector[0], iteration_vector[1], iteration_vector[2], iteration_vector[3], iteration_vector[4]);
				if(i == freeze_circle)
               	{
               		freeze_op->freeze_circle(iteration_vector[0], iteration_vector[1], iteration_vector[2], iteration_vector[3], iteration_vector[4], 0.8);
               	}
			}

			// Y changes
			if(temp_circle)
			{
				addtemp_op->calculate_operator(iteration_vector[0]);
				//addtemp_op->calculate_operator(iteration_vector[2]);
				
				if(i == temp_circle)
				{	//@ first iteration mask is prepared
					addtemp_op->temp_rectangle(iteration_vector[0], 0.1);
				}
			}
            // Special: Freeze circle
            /*if(freeze_circle){
                freeze_op->calculate_operator(iteration_vector[0], iteration_vector[1], iteration_vector[2], iteration_vector[3], iteration_vector[4]);
                if(i == freeze_circle) {
                    freeze_op->freeze_circle(iteration_vector[0], iteration_vector[1], iteration_vector[2], iteration_vector[3], iteration_vector[4], 0.8);
                }
            }

            //Special: Temp circle
           if(temp_circle) {
              mask_op->calculate_operator(iteration_vector[0]);
             if(i == temp_circle) {
                 mask_op->temp_circle(iteration_vector[0], 0.8);
             }
           }*/

            #ifdef TIMER
                stepTimer.Start();
            #endif

            //DBGOUT("Start timestep");
            stepper->step_time(iteration_vector, delta_t);
            //DBGOUT("Finish timestep");

            //force the signal to be real
            make_real_signal(iteration_vector);

            #ifdef TIMER
                stepTimer.Stop();
            #endif
        }

        /***********************************************************************************/
        /** Trace particles                                                               **/
        /***********************************************************************************/
        #ifdef USE_PARTICLES
            #ifdef TIMER
                partTimer.Start();
            #endif
            //DBGOUT("Step particles");

            // Prepare variables that are used as if only one tracer was there
            for(int tracer_i = 0; tracer_i < tracer_count; ++tracer_i) {
                particle_tracer* tracer = tracers[tracer_i];

                //DBGOUT("Start particle step");
                tracer->step_particles(vel_op_init, f_folder, g_folder, F_folder, G_folder, delta_t);

                //...transfer the particle positions to disk
                if(tracer->is_buffer_full()){
                    cout << "read out particle positions" << endl;
                    matrix_folder_real* particle_positions = tracer->get_particle_positions(particle_tracer::Clear);
                    if(write_tracks) {
                        trackwriters.at(tracer_i)->append(particle_positions);
                    }
                    delete particle_positions;
                }

                if(calc_secondmoment) {
                    //DBGOUT("Calc particle second moment");
                    CUDA_FLOAT_REAL M2 = tracer->calc_secondmoment();
                    second_moment_writers[tracer_i]->append(i * delta_t, M2);
                    if(second_moment_writers[tracer_i]->is_buffer_full()) {
                        second_moment_writers[tracer_i]->write_buffer_to_file();
                    }
                }

            }

            #ifdef TIMER
                partTimer.Stop();
            #endif
        #endif  // USE_PARTICLES


        /***********************************************************************************/
        /** Update server                                                                 **/
        /***********************************************************************************/
        #ifdef TIMER
            outputTimer.Start();
        #endif
        #ifdef USE_SERVER
        // Send temperature map to server
		serverWriter.sendData(i, &netUserInput);
		//printf("Nt = %d, mouse at %f  %f, changing field %d, modif factor %.2f\n", i, netUserInput.relPosX, netUserInput.relPosY, netUserInput.mode, netUserInput.temperature);
		printf("Nt = %d, mouse at %f  %f, changing field %d, modif factor %.2f\n", i, netUserInput.posX, netUserInput.posY, netUserInput.mode, netUserInput.temp);
		
		if(netUserInput.mode == 1) // temperature
		{
			//interactionMask->set_mask_parameters(netUserInput.relPosX, netUserInput.relPosY, netUserInput.radius, netUserInput.temperature);
			interactionMask->set_mask_parameters(netUserInput.posX, netUserInput.posY, netUserInput.radius, netUserInput.temp);
			interactionMask->masker(iteration_vector[0]);
		}
		//printf("%d\n", i);
        #endif
		//m changes
		/*apply mask on :
			temperature field : iteration_vector[0]
			f field : iteration_vector[1]
			g field : iteration_vector[2]
		*/

        /***********************************************************************************/
        /** Write binary files                                                            **/
        /***********************************************************************************/
		/// For speed purpose, commented        
		if(write_file > 0 && i % write_file == 0) {
            cout << "writing tmp-files to disk!" << endl;
            //...folders in fourier space (binary data)
            sprintf(tmp_filename, "theta_%08d.bin", i);
            matrix_folder_writer::write_binary_file(tmp_filename, theta_folder);
            //...folders in fourier space (binary data)
            sprintf(tmp_filename, "f_%08d.bin", i);
            matrix_folder_writer::write_binary_file(tmp_filename, f_folder);
            //...folders in fourier space (binary data)
            sprintf(tmp_filename, "g_%08d.bin", i);
            matrix_folder_writer::write_binary_file(tmp_filename, g_folder);
            //...folders in fourier space (binary data)
            sprintf(tmp_filename, "F_%08d.bin", i);
            matrix_folder_writer::write_binary_file(tmp_filename, F_folder);
            //...folders in fourier space (binary data)
            sprintf(tmp_filename, "G_%08d.bin", i);
            matrix_folder_writer::write_binary_file(tmp_filename, G_folder);
            
            /*sprintf(tmp_filename, "theta_%08d.gnu", i); // Debugging only
            matrix_folder_writer::write_gnuplot_file(tmp_filename, theta_folder);*/
            
            cout << "wrote file" << endl;//

            #ifdef USE_PARTICLES
                if(write_tracks) {
                    for(int tracer_i = 0; tracer_i < tracer_count; ++tracer_i) {
                        sprintf(tmp_filename, "particles_le%.0le_%08d.trkss", lewis_numbers.at(tracer_i), i);
                        trackwriters.at(tracer_i)->write_file(tmp_filename);
                    }
                }
            #endif  // USE_PARTICLES
        }

/*    #define WRITE_SNAPSHOT
        #ifdef WRITE_SNAPSHOT
                //...temperature
                sprintf(tmp_rundescr, "Midplane Temperature.  Simulation: Ra=%.1lf Pr=%.1lf Size=%.1lfx%.1lfx1.0  Technical: Modes=%dx%dx%d dt=%.3lf  Time: T=%.3lf ",
                        rayleigh_number, prandtl_number, cube_length_x, cube_length_y,
                        M, N, L, delta_t, i*delta_t);
                sprintf(tmp_filename, "temperature_%08d.dat", i);
                vector<CUDA_FLOAT_REAL> at_z(1, 0.0);
                matrix_folder_real* temp_tmp = temp_op_init->calculate_operator_at(theta_folder, at_z);
                matrix_folder_writer::write_binary_3d_layer(tmp_filename, temp_tmp, tmp_rundescr);
                sprintf(tmp_filename, "temperature_%08d.txt", i);
                matrix_folder_writer::write_gnuplot_3d_layer(tmp_filename, temp_tmp, tmp_rundescr);
                delete temp_tmp;
        
                 Write max vel
                {
                    matrix_folder_real* vel_tmp = vel_op_init->calculate_operator(f_folder, g_folder, F_folder, G_folder, 5);
                    cout << "max = " << matrix_max(vel_tmp) << endl;
                    delete vel_tmp;
                }

                cout << "wrote snapshot" << endl;
            }
        #endif
*/
        #ifdef TIMER
            outputTimer.Stop();
        #endif

        /***********************************************************************************/
        /** Update time                                                                   **/
        /***********************************************************************************/
/// For speed purpose, commented  

       // fprintf(stdout, "iteration: %d time: %.3lf\n", i, i*delta_t);
        #ifdef TIMER
       //     fprintf(stdout, "Timer: step: %.2lf ms, part: %.2lf ms, output: %.2lf ms\n",
                  //  1000 * stepTimer.Elapseds() / i, 1000 * partTimer.Elapseds() / i, 1000 * outputTimer.Elapseds() / i);
        #endif
        fflush(stdout);//*/

        // Check for errors in time loop
        DBGSYNC();
    }
    delete temp_op_init;
    delete vel_op_init;    

    /***********************************************************************************/
    /** Write final files                                                             **/
    /***********************************************************************************/
    /// For speed purpose, commented  

    /*matrix_folder_writer::write_binary_file("theta_result.bin", theta_folder);
    matrix_folder_writer::write_binary_file("f_result.bin", f_folder);
    matrix_folder_writer::write_binary_file("g_result.bin", g_folder);
    matrix_folder_writer::write_binary_file("F_result.bin", F_folder);
    matrix_folder_writer::write_binary_file("G_result.bin", G_folder);*/

    #ifdef WRITE_VTK
        //write out velocity
        calculate_velocity_operator* vel_op = calculate_velocity_operator::init(dimensions, cube_length);
        matrix_folder_real* velocity = vel_op->calculate_operator(f_folder, g_folder, F_folder, G_folder);
        matrix_folder_writer::write_vtk_file("velocity.vtk", velocity, 1);    // Compile will fail: this only works for complex matrices
        delete velocity;
        delete vel_op;

        //write out temperature
        calculate_temperature_operator* temp_op = calculate_temperature_operator::init(dimensions, cube_length);
        matrix_folder_real* temperature = temp_op->calculate_operator(theta_folder);
        matrix_folder_writer::write_vtk_file("temperature.vtk", temperature, 1);
        delete temperature;
        delete temp_op;
    #endif

    #ifdef USE_PARTICLES
        if(write_tracks) {
            for(int tracer_i = 0; tracer_i < tracer_count; ++tracer_i) {

                cout << "read out particle positions" << endl;
                matrix_folder_real* particle_positions = tracers[tracer_i]->get_particle_positions(particle_tracer::Clear);
                trackwriters[tracer_i]->append(particle_positions);
                delete particle_positions;

                sprintf(tmp_filename, "particles_le%.0le_%08d.trkss", lewis_numbers.at(tracer_i), number_of_iterations);
                trackwriters[tracer_i]->write_file(tmp_filename);
            }
        }
    #endif  // USE_PARTICLES

#ifdef CALC_STRUCTFN
    //calculate structure function in x direction
    matrix_folder* struct_function = structure_op->calculate_temperature_structure_function_x(theta_folder, 0.0, cube_length_x, M);
    matrix_folder_writer::write_gnuplot_vector_file("structure_function_2d_x_direction.dat", struct_function);
    cout << "calculating 2nd-order-2D-structure in x direction function finished!" << endl;

    //calculate structure function in y direction
    struct_function = structure_op->calculate_temperature_structure_function_y(theta_folder, 0.0, cube_length_y, N);
    matrix_folder_writer::write_gnuplot_vector_file("structure_function_2d_y_direction.dat", struct_function);
    cout << "calculating 2nd-order-2D-structure in y direction function finished!" << endl;
#endif

    // Write second moment to file
#ifdef USE_PARTICLES
    for(size_t tracer_i = 0; tracer_i < second_moment_writers.size(); ++tracer_i) {
        second_moment_writers[tracer_i]->write_buffer_to_file();
    }
#endif


    /***********************************************************************************/
    /** Free memory                                                                   **/
    /***********************************************************************************/
    // Clear Multitracer storage
#ifdef USE_PARTICLES
    for(int tracer_i = 0; tracer_i < tracer_count; ++tracer_i) delete tracers[tracer_i];
    tracers.clear();

    for(size_t i = 0; i < trackwriters.size(); ++i) delete trackwriters[i];
    trackwriters.clear();

    for(size_t i = 0; i < second_moment_writers.size(); ++i) delete second_moment_writers[i];
    second_moment_writers.clear();
#endif  // USE_PARTICLES

    delete stepper;
    delete theta_folder;
    delete f_folder;
    delete g_folder;
    delete F_folder;
    delete G_folder;

    cout << "finished simulation\n" << endl;

    return EXIT_SUCCESS;
}










