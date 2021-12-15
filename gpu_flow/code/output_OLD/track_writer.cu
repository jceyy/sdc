#include "track_writer.h"


/***********************************************************************************/
/** track_writer                                                                  **/
/***********************************************************************************/
track_writer::track_writer(const char* description, int _write_period) {

    write_period = _write_period;
    time = 0;
    offset = 0;


    movie.data.description = description;
    movie.data.fs = 1;
}

track_writer::~track_writer() {
}

void track_writer::append(matrix_folder_real *F) {

    matrix_host_real xPos(F->get_matrix(0), matrix_host_real::Copy);
    matrix_host_real yPos(F->get_matrix(1), matrix_host_real::Copy);
    matrix_host_real zPos(F->get_matrix(2), matrix_host_real::Copy);

    // If first append: Set basic info
    if(movie.get_NbFrames() == 0) {
        movie.data.num_tracks = xPos.get_matrix_size(0);
    }

    int matrix_num_frames = xPos.get_matrix_size(1);

    cout << "At time "<<time<<": Append up to "<<matrix_num_frames<<" frames (offset"<<offset<<")."<<endl;

    // Error checks
    if((unsigned int)xPos.get_matrix_size(0) != movie.data.num_tracks) EXIT_ERROR("Number of tracks mismatch.");
    if((unsigned int)yPos.get_matrix_size(0) != movie.data.num_tracks) EXIT_ERROR("Number of tracks mismatch.");
    if((unsigned int)zPos.get_matrix_size(0) != movie.data.num_tracks) EXIT_ERROR("Number of tracks mismatch.");
    if(yPos.get_matrix_size(1) != matrix_num_frames) EXIT_ERROR("Number of timesteps mismatch.");
    if(zPos.get_matrix_size(1) != matrix_num_frames) EXIT_ERROR("Number of timesteps mismatch.");

    // Append matrices to movie
    track_movie_frame frame;
    vector<int> indices(2);
    for(int t = offset; t < matrix_num_frames; t += write_period) {

        // Generate frame
        frame.time = time;
        for(unsigned int p = 0; p < movie.data.num_tracks; ++p) {
            indices[0] = p;
            indices[1] = t;
            frame.x.push_back(xPos.get_entry(indices));
            frame.y.push_back(yPos.get_entry(indices));
            frame.z.push_back(zPos.get_entry(indices));
        }
        movie.data.frames.push_back(frame);
        frame.clear();

        time += write_period;
        offset = t - matrix_num_frames + write_period;
    }
}

// Write file clears current data, but leaves time unchanged for next append
void track_writer::write_file(char *filename) {

    // Only write if data is available
    if(movie.get_NbFrames() == 0) return;

    ofstream of(filename, ios::binary);
    if(!of.is_open()) {
        cerr << "Open file failed: " << filename << endl;
        return;
    }

    movie.Write(of);
    movie.clear();
    of.close();
}






