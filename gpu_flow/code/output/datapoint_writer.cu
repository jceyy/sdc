#include "datapoint_writer.h"


datapoint_writer::datapoint_writer(const char* filename, unsigned int pbuffer_size, const char* titlestring)
    : _buffer_size(pbuffer_size), _data_index(0) {

    strncpy(_filename, filename, FILENAME_LEN);
    strncpy(_titlestring, titlestring, 128);

    _time = new CUDA_FLOAT_REAL[_buffer_size];
    _vals = new CUDA_FLOAT_REAL[_buffer_size];

    write_header();
}

datapoint_writer::~datapoint_writer() {
    write_buffer_to_file();

    delete [] _time;
    delete [] _vals;
}

void datapoint_writer::write_header(){

    //...open the file and check if opened
    ofstream of(_filename, ios_base::out);

    if(!of.is_open()) EXIT_ERROR("Open file failed.");

    of << "# Data point writer\n";
    of << "# " << _titlestring << '\n';
    of << "# Data ordering: t, val" << endl;

    // Close filestream
    of.close();
}

void datapoint_writer::write_data(){


    //...open the file and check if opened
    ofstream of(_filename, ios_base::app);
    if(!of.is_open()) EXIT_ERROR("Open file failed.");

    of.precision(6);
    //of.setf(ios::scientific);

    for(unsigned int i = 0; i < _data_index; i++) {
        of << _time[i]  << '\t' << _vals[i] << endl;
    }

    // Close filestream
    of.close();
}

void datapoint_writer::append(CUDA_FLOAT_REAL time, CUDA_FLOAT_REAL val) {

    if(_data_index >= _buffer_size) EXIT_ERROR("Buffer full.");

    _time[_data_index] = time;
    _vals[_data_index] = val;

    _data_index++;
}

void datapoint_writer::write_buffer_to_file() {
    if(_data_index > 0) {
        write_data();
        _data_index = 0;
    }
}

bool datapoint_writer::is_buffer_full() {
    return (_data_index >= _buffer_size);
}





