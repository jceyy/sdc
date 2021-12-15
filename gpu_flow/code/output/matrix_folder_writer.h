#ifndef MATRIX_FOLDER_WRITER_H
#define MATRIX_FOLDER_WRITER_H

//system includes
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

//my includes
#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"
#include "../matrix/matrix_host_iterator.h"

class matrix_folder_writer {

private:

protected:

public:
    enum App { append, noappend };
    static void write_vtk_file(string filename, matrix_folder* folder, int type);
	static void write_gnuplot_file(string filename, matrix_folder* folder);
    static void write_gnuplot_file(string filename, matrix_folder* folder, int matrix_i);
    static void write_binary_3d_layer(string filename, matrix_folder_real* folder, const char *description,
                                      const int layer = 0);
    static void convert_binary_3d_layer(const char *filename_in, const char *filename_out);
    static void write_gnuplot_3d_layer(string filename, matrix_folder_real* folder, const char* description,
                                       const int layer = 0);
    static void write_gnuplot_vector_file(string filename, matrix_folder* folder);
    static void write_gnuplot_vector_file(string filename, matrix_folder_real* folder);
    static void write_gnuplot_vector_file_2d(string filename, matrix_folder_real* folder,
                                             const char* description,
                                             int select_dim, int select_val);
    static void write_gnuplot_particle_file(string filename, matrix_folder* folder, int starttime, App as,
                                            int write_interval, double dt);


	static void write_binary_file(string filename, matrix_folder* folder);
    static matrix_folder* read_binary_file(string filename);

    static void write_binary_track_file(string filename, matrix_folder* folder, int starttime, App as,
                                        int write_interval, double dt);
    static void read_binary_track_file(string filename);
};

#endif

