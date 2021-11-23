#include <iostream>
#include <string>
#include "output/matrix_folder_writer.h"
using namespace std;

int main(int argc, char** argv) {

    if(argc < 3) {
        cerr << "Usage: ./convert <infile> <outfile>" << endl;
        return 0;
    }

    matrix_folder_writer::convert_binary_3d_layer(argv[1], argv[2]);
}
