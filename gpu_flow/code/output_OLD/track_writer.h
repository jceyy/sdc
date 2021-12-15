#ifndef TRACK_WRITER_H
#define TRACK_WRITER_H

//system includes
#include <vector>
#include <iostream>
#include <fstream>
#include <stdint.h>
using namespace std;

//my includes
#include "../cuda_defines.h"
#include "../matrix/matrix_folder.h"
#include "track_movie.h"

class track_writer {
private:
    track_movie movie;
    // Write period
    int write_period;
    // End time of last append:
    int time;
    // Start frame of next append
    int offset;
public:
    track_writer(const char* description, int _write_period = 1);
    ~track_writer();
    void append(matrix_folder_real* F);
    void write_file(char* filename);
};

#endif

