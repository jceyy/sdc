// This program reads a .trk file, and prints the distribution of
// the particles at a given time to an output file.

#include "util/inputparser.h"

//define TRACK
#ifdef TRACK
#include "../../eulerview/eulerview_v123/Mov.cpp"
typedef TrackMov MovType;
#else
#include "output/track_movie.h"
typedef track_movie MovType;
#endif

#define FILENAME_LEN 128
using namespace std;

int xtobin(double x, double xmin, double xmax, int nbins);
double bintox(int bin, double xmin, double xmax, int nbins);
void extract_distribution(vector<unsigned int>& bins, unsigned int& npoints, MovType& movie, int nbins_x, int nbins_y, vector<double>& limits, int time, std::vector<int>& select_dim);

int main(int argc, char** argv) {

    /***********************************************************************************/
    /** Read command line arguments                                                   **/
    /***********************************************************************************/
    inputparser ip(argc, argv);

    // time index
    int time = 0;
    ip.get<int>(time, "-time", inputparser::mandatory);

    // Get the path of input files
    int num_infiles;
    if(ip.getopt("-ninfiles")) {
        ip.get<int>(num_infiles, "-ninfiles", inputparser::mandatory);
    } else {
        num_infiles = 1;
    }
    char** infilenames = new char*[num_infiles];
    for(int i = 0; i < num_infiles; ++i) {
        infilenames[i] = new char[FILENAME_LEN];
    }
    ip.getstrings(infilenames, FILENAME_LEN, 1, "-infile", inputparser::mandatory);

    // the path of output file
    char** outfilenames = new char*[1];
    outfilenames[0] = new char[FILENAME_LEN];
    ip.getstrings(outfilenames, FILENAME_LEN, 1, "-outfile", inputparser::mandatory);

    // number of bins in x- and y-direction
    std::vector<int> nbins(2,0);
    ip.get<int>(nbins, 2, "-nbins", inputparser::mandatory);
    int nbins_x = nbins.at(0), nbins_y = nbins.at(1);

    // edge length in x- and y-direction
    std::vector<double> cube_length(2,1.0);
    ip.get<double>(cube_length, 2, "-cube_length", inputparser::mandatory);
    double cube_length_x = cube_length.at(0), cube_length_y = cube_length.at(1);

    // limits of sampling (xmin xmax ymin ymax)
    std::vector<double> limits(4);
    limits.at(0) = 0; limits.at(1) = cube_length_x; limits.at(2) = 0; limits.at(3) = cube_length_y;
    ip.get<double>(limits, 4, "-limits", inputparser::optional);

    // Sample dimensions (x 0, y 1, z 2)
    std::vector<int> select_dim(2);
    select_dim.at(0) = 0; select_dim.at(1) = 1;
    ip.get<int>(select_dim, 2, "-select_dim", inputparser::optional);

    MovType movie;

    // Read input files
    for(int i = 0; i < num_infiles; ++i) {
        ifstream infile(infilenames[i], ios::binary);
        if(!infile.is_open()) {
            cerr << "Open file failed: " << infilenames[i] << endl;
            return 1;
        }
        cout << "Load file " << infilenames[i] << endl;

#ifdef TRACK
        movie.Read(infile);
#else
        movie.read_append(infile);
#endif
        infile.close();
    }

    if(movie.empty()) {
        cout << "No input read. Terminating." << endl;
        return 0;
    }

    //movie.print();
    cout << "Lookup time "<<time<<" in ["<<movie.get_start(0)<<":"<<movie.get_end(0)<<"]\n";

    // Create bins
    vector<unsigned int> bins(nbins_x*nbins_y,0);
    unsigned int npoints = 0;

    // THE FUNCTION!!!
    extract_distribution(bins, npoints, movie, nbins_x, nbins_y, limits, time, select_dim);


    // Get Maximum
    unsigned int max = 0;
    for(size_t i = 0; i < bins.size(); ++i) {
        if(bins.at(i) > max) max = bins.at(i);
    }

    // Print map to output
    ofstream outfile(outfilenames[0]);

    outfile << "# Data from "<<npoints<<" points at time "<<time<< " from file "<<infilenames[0]<< endl;
    outfile << "# Max " << (max / double(npoints)) << "\n";

    // Print 10 particle positions
    for(unsigned long fr = 0; fr < movie.get_NbFrames(); ++fr) {
        int frame_time = movie.data.frames.at(fr).time;
        if(frame_time == time) {
            unsigned long trmax = 10;
            if(movie.get_NbTracks() < trmax) trmax = movie.get_NbTracks();
            for(unsigned long tr = 0; tr < trmax; ++tr) {
                double pos[3];
                movie.getPos(fr, tr, pos[0], pos[1], pos[2]);
                outfile << "# " << pos[0] << '\t' << pos[1] << '\t' << pos[2] << '\n';
            }
        }
    }

    // Print the bins
    for(int bin_x = 0; bin_x < nbins_x; ++bin_x) {
        for(int bin_y = 0; bin_y < nbins_y; ++bin_y) {
            outfile << bintox(bin_x, limits.at(0),limits.at(1),nbins_x) << '\t' <<
                       bintox(bin_y, limits.at(2),limits.at(3),nbins_y) << '\t' <<
                       double(bins.at(bin_x + nbins_x * bin_y)) / double(npoints) << '\n';
        }
        outfile << '\n';
    }
    outfile.close();
    return 0;
}

void extract_distribution(vector<unsigned int>& bins, unsigned int& npoints, MovType& movie, int nbins_x, int nbins_y, vector<double>& limits, int time, vector<int>& select_dim) {

    // Find correct frame
    for(unsigned long fr = 0; fr < movie.get_NbFrames(); ++fr) {
        int frame_time = movie.data.frames.at(fr).time;
        if(frame_time /*>=*/== time) {
            if(frame_time != time) {
                cout << "Time mismatch: t="<<time
                     <<", look at frame "<<fr<<" (time "<<frame_time<<")"<<endl;
            }

            cout << "Extract "<<movie.get_NbTracks()<<" points at time "<<frame_time<<endl;
            for(unsigned long tr = 0; tr < movie.get_NbTracks(); ++tr) {
                double pos[3];
                if(movie.getPos(fr, tr, pos[0], pos[1], pos[2])) {

                    // Convert position to bin
                    double x = pos[select_dim.at(0)];
                    double y = pos[select_dim.at(1)];
                    int bin_x = xtobin(x, limits.at(0), limits.at(1), nbins_x);
                    int bin_y = xtobin(y, limits.at(2), limits.at(3), nbins_y);
                    if(!(bin_x < nbins_x)) {
                        cout << "Point "<<x<<","<<y<<" not in grid: At xbin "<<bin_x<<endl;
                        continue;
                    }
                    if(!(bin_x >= 0)) {
                        cout << "Point "<<x<<","<<y<<" not in grid: At xbin "<<bin_x<<endl;
                        continue;
                    }
                    if(!(bin_y < nbins_y)) {
                        cout << "Point "<<x<<","<<y<<" not in grid: At ybin "<<bin_y<<endl;
                        continue;
                    }
                    if(!(bin_y >= 0)) {
                        cout << "Point "<<x<<","<<y<<" not in grid: At ybin "<<bin_y<<endl;
                        continue;
                    }

                    // Store position to probability distribution map (row major)
                    bins.at(bin_x + nbins_x * bin_y)++;
                    npoints++;
                }
            }
            //break;
        }
    }
}

int xtobin(double x, double xmin, double xmax, int nbins) {
    return ((x-xmin)*nbins) / (xmax-xmin);
}

double bintox(int bin, double xmin, double xmax, int nbins) {
    return xmin + (bin * (xmax-xmin))/nbins;
}




