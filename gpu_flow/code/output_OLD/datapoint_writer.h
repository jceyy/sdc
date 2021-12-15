#ifndef DATAPOINT_WRITER_H
#define DATAPOINT_WRITER_H

//system includes
#include <vector>
#include <iostream>
#include <fstream>
#include <stdint.h>
using namespace std;

//my includes
#include "../cuda_defines.h"

class datapoint_writer {

private:
    unsigned int _data_index;
    unsigned int _buffer_size;
    CUDA_FLOAT_REAL* _time;
    CUDA_FLOAT_REAL* _vals;
    char _filename[FILENAME_LEN];
    char _titlestring[128];

    void write_header();
    void write_data();
protected:

public:
    datapoint_writer(const char* filename, unsigned int pbuffer_size, const char* titlestring = "");
    ~datapoint_writer();

    void append(CUDA_FLOAT_REAL time, CUDA_FLOAT_REAL val);
    void write_buffer_to_file();

    bool is_buffer_full();
};

#endif
