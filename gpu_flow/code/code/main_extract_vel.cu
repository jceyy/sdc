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
//include "util/init.h"
#include "util/util.h"
#include "util/inputparser.h"
#include "matrix/matrix_host.h"
#include "matrix/matrix_device.h"
#include "matrix/matrix_folder.h"
#include "output/matrix_folder_writer.h"
#include "output/datapoint_writer.h"
#include "operator/calculate_velocity_operator.h"
#include "operator/calculate_temperature_operator.h"


int main(int argc, char** argv) {

    /***********************************************************************************/
    /** Read command line arguments                                                   **/
    /***********************************************************************************/
    inputparser ip(argc, argv);
    ip.print();

    //...the cuda device
    int cuda_device = 0;
    ip.get<int>(cuda_device, "-device", inputparser::optional);

    // ...cube length in x and y direction
    // Length of the cube in physical space as multiples of the vertical length, which is equal to one
	vector<CUDA_FLOAT_REAL> cube_length;
    ip.get<CUDA_FLOAT_REAL>(cube_length, 2, "-clength", inputparser::mandatory);
    //...z length is fixed to 1 because of special ansatz functions
    cube_length.resize(3); cube_length.at(2) = 1.0;

    //...input datafiles
    char infilename_theta[FILENAME_LEN]; strcpy(infilename_theta, "");
    char infilename_f[FILENAME_LEN];     strcpy(infilename_f, "");
    char infilename_g[FILENAME_LEN];     strcpy(infilename_g, "");
    char infilename_F[FILENAME_LEN];     strcpy(infilename_F, "");
    char infilename_G[FILENAME_LEN];     strcpy(infilename_G, "");
    char *infiles[5] = { infilename_theta, infilename_f, infilename_g, infilename_F, infilename_G };
    ip.getstrings(infiles, FILENAME_LEN, 5, "-files", inputparser::mandatory);

    // Number of z points to write for vel (optional, 0: no file written)
    int write_vel_nz = 0;
    ip.get<int>(write_vel_nz, "-write_vel_nz", inputparser::optional);
    char outfilename_vel[FILENAME_LEN]; strcpy(outfilename_vel, "");
    if(write_vel_nz > 0) {
        char *outfile_vel[1] = {outfilename_vel};
        ip.getstrings(outfile_vel, FILENAME_LEN, 1, "-outfile_vel", inputparser::mandatory);
    }

    // Number of z points to write for temp (optional, 0: no file written)
    int write_temp_nz = 0;
    ip.get<int>(write_temp_nz, "-write_temp_nz", inputparser::optional);
    char outfilename_temp[FILENAME_LEN]; strcpy(outfilename_temp, "");
    if(write_temp_nz > 0) {
        char *outfile_temp[1] = {outfilename_temp};
        ip.getstrings(outfile_temp, FILENAME_LEN, 1, "-outfile_temp", inputparser::mandatory);
    }

    // Characteristic velocity scale
    int write_charvel_nz = 0;
    ip.get<int>(write_charvel_nz, "-write_charvel_nz", inputparser::optional);
    char outfilename_charvel[FILENAME_LEN]; strcpy(outfilename_charvel, "");
    if(write_charvel_nz > 0) {
        ip.getstring(outfilename_charvel, FILENAME_LEN, "-outfile_charvel", inputparser::mandatory);
    }

    // Restrict dimension select_dim[0] to value select_dim[1]
    vector<int> select_dim(2,0);
    if(write_vel_nz > 0 || write_temp_nz > 0) {
        ip.get<int>(select_dim, 2, "-select_dim", inputparser::mandatory);
    }

    // Description
    char description[FILENAME_LEN]; sprintf(description, "");
    ip.getstring(description, FILENAME_LEN, "-description", inputparser::optional);

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
    matrix_folder* theta_folder;
    matrix_folder* f_folder;
    matrix_folder* g_folder;
    matrix_folder* F_folder;
    matrix_folder* G_folder;

    // Read files if all filenames are present
    // Read binary input files

    cout << "...reading init data from binary file" << endl;

    theta_folder = matrix_folder_writer::read_binary_file(infilename_theta);

    // Get number of modes in Fourier space
    vector<int> dimensions(theta_folder->get_matrix(0)->get_matrix_dimension());
    int M = 2*(dimensions.at(0)-1);
    int N = dimensions.at(1);
    int L = dimensions.at(2);
    cout << "number of spectral coefficients: " << M/2+1 << " x " << N << " x " << L << endl;

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


    /***********************************************************************************/
    /** Extract information                                                           **/
    /***********************************************************************************/
    if(write_vel_nz > 0) {
        //write out velocity
        calculate_velocity_operator* vel_op = calculate_velocity_operator::init(dimensions, cube_length);
        matrix_folder_real* velocity = vel_op->calculate_operator(f_folder, g_folder, F_folder, G_folder, write_vel_nz);
        matrix_folder_writer::write_gnuplot_vector_file_2d(string(outfilename_vel), velocity, description, select_dim.at(0), select_dim.at(1));
        delete velocity;
        delete vel_op;
    }

    if(write_temp_nz > 0) {
        //write out temperature
        calculate_temperature_operator* temp_op = calculate_temperature_operator::init(dimensions, cube_length);
        matrix_folder_real* temperature = temp_op->calculate_operator(theta_folder, write_temp_nz);
        matrix_folder_writer::write_gnuplot_vector_file_2d(string(outfilename_temp), temperature, description, select_dim.at(0), select_dim.at(1));
        delete temperature;
        delete temp_op;
    }

    if(write_charvel_nz > 0) {
        //write out characteristic velocity (which is defined as <|u(x)|> )
        cout << "Extract characteristic velocity U on " << write_charvel_nz << " levels." << endl;

        // Calculate velocity
        calculate_velocity_operator* vel_op = calculate_velocity_operator::init(dimensions, cube_length);
        matrix_folder_real* velocity = vel_op->calculate_operator(f_folder, g_folder, F_folder, G_folder, 10);

        // Extract matrices
        matrix_host_real* vel_host[3];
        matrix_host_real_iterator* it_host[3];
        for(int i = 0; i < 3; i++) {
            vel_host[i] = new matrix_host_real(velocity->get_matrix(i), matrix_host_real::Copy);
            it_host[i] = matrix_host_real_iterator::create_iterator(vel_host[i]);
        }

        // Calculate average
        double count = 0.0, avg = 0.0;
        while(it_host[0]->has_next() && it_host[1]->has_next() && it_host[2]->has_next()) {

            double val = 0.0;
            for(int i = 0; i < 3; i++) {
                double v = it_host[i]->next();
                val += v*v;
            }

            // Avoid NaNs
            if(val == val) {
                count++;
                avg += sqrt(val);
            }
        }
        avg /= count;
        cout << "Averaged " << count << " points: U="<<avg<<endl;

        // Output result
        datapoint_writer w(outfilename_charvel, 1, description);
        w.append(0.0, avg);
        w.write_buffer_to_file();

        // Free memory
        delete velocity;
        delete vel_op;
        for(int i = 0; i < 3; i++) {
            delete vel_host[i];
            delete it_host[i];
        }
    }


	delete theta_folder;
	delete f_folder;
	delete g_folder;
	delete F_folder;
	delete G_folder;

    cout << "finished write\n" << endl;

    return EXIT_SUCCESS;
}








