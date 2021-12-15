// This is the file main_convert.cu, just with inlined
// function from matrix_folder_writer, so that CUDA is
// not needed to complete the mission.


#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <stdlib.h>
//include "output/matrix_folder_writer.h"
using namespace std;

void convert_binary_3d_layer(const char* filename_in, const char* filename_out);

int main(int argc, char** argv) {

    if(argc < 3) {
        cerr << "Usage: ./convert <infile> <outfile>" << endl;
        return 0;
    }

    convert_binary_3d_layer(argv[1], argv[2]);
}


void convert_binary_3d_layer(const char* filename_in, const char* filename_out) {
    
    
    // Open the input file
    ifstream infile(filename_in, ios::in | ios::binary);
    if(!infile.is_open()) {
        cerr << "convert_binary_3d_layer: Open file failed: " << filename_in << endl;
        return;
    }
    
    // Open the output file
    ofstream of(filename_out);
    if(!of.is_open()) {
        cerr << "convert_binary_3d_layer: Open file failed: " << filename_out << endl;
        return;
    }
    
    // Read description from file
    int len = 0;
    infile.read((char*) &len, sizeof(int));
    if(len < 0 || len > 1024) {
        cerr << "convert_binary_3d_layer: Read failed: len = " << len << endl;
        return;
    }
    
    char* description = new char[len];
    infile.read(description, sizeof(char)*len);
    
    // Read information about the size of the matrix
    int dim_x = 0;
    int dim_y = 0;
    infile.read((char*) &dim_x, sizeof(int));
    infile.read((char*) &dim_y, sizeof(int));
    if(dim_x < 1 || dim_x > 32000 || dim_y < 1 || dim_y > 32000) {
        cerr << "convert_binary_3d_layer: Read failed: dim_x dim_y = " << dim_x << " " << dim_y << endl;
        return;
    }
    
    // Read in the data
    float* data = new float[dim_x*dim_y];
    infile.read((char*) data, sizeof(float) * dim_x*dim_y);
    
    // Read complete
    cout << "Convert " << dim_x << "x" << dim_y << " floats with description " << description << endl;
    
    // Write out the data
    of << "# " << description << endl;
    of << "# data ordering: x,y,.... ,real(data_x)" << endl;
    of.precision(4);
    of.setf(ios::scientific);
    
    for(int y = 0; y < dim_y; ++y) {
        of << '\n';
        for(int x = 0; x < dim_x; ++x) {
            of << x << '\t' << y << '\t' << data[x+y*dim_x] << '\n';
        }
    }
    
    // Delete memory
    delete [] description;
    delete [] data;
    
    //close file
    infile.close();
    of.close();
}
