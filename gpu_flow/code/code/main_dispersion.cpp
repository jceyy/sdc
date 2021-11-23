// This program reads a .trk file, and prints the distribution of
// the particles at a given time to an output file.

#include <vector>
#include "util/inputparser.h"
#include "output/track_movie.h"

typedef track_movie MovType;

#define FILENAME_LEN 256
using namespace std;

void extract_R2(MovType& movie, vector<double>& cube_length, double dt, fstream& writer);
void extract_bin(MovType& movie, vector<double>& cube_length, double dt, double mindiff, double maxdiff, fstream& writer);
void extract_rprofile(MovType& movie, vector<double>& cube_length, double dt, int rprofile_index, int nbins, fstream& R2writer);
int xtobin(double x, double xmin, double xmax, int nbins);
double bintox(int bin, double xmin, double xmax, int nbins);

int main(int argc, char** argv) {

    /***********************************************************************************/
    /** Read command line arguments                                                   **/
    /***********************************************************************************/
    inputparser ip(argc, argv);

    // Get the path of input files
    int ninfiles = 1;
    ip.get<int>(ninfiles, "-ninfiles", inputparser::optional);
    if(ninfiles < 1 || ninfiles > 1000) return 1;
    char** infiles = new char*[ninfiles];
    for(int i = 0; i < ninfiles; ++i) infiles[i] = new char[FILENAME_LEN];
    ip.getstrings(infiles, FILENAME_LEN, ninfiles, "-infiles", inputparser::mandatory);

    // the path of output file
    char outfilename[FILENAME_LEN];
    ip.getstring(outfilename, FILENAME_LEN, "-outfile", inputparser::mandatory);

    // edge length in x- and y-direction
    std::vector<double> cube_length(2,1.0);
    ip.get<double>(cube_length, 2, "-cube_length", inputparser::mandatory);

    double dt = 0.;
    ip.get<double>(dt, "-dt", inputparser::mandatory);

    bool r0bin = ip.getopt("-r0bin");
    std::vector<double> r0bin_minmax(2,0.0);
    if(r0bin) {
        ip.get<double>(r0bin_minmax, 2, "-r0bin", inputparser::mandatory);
        if(!(r0bin_minmax.at(1)>r0bin_minmax.at(0))) {
            cerr << "Bin max must be larger than bin min!" << endl;
            return -1;
        }
    }

    bool rprofile = ip.getopt("-rprofile");
    int rprofile_index = 0;
    int rprofile_nbins = 1000;
    if(rprofile) {
        ip.get<int>(rprofile_index, "-rprofile", inputparser::mandatory);
        ip.get<int>(rprofile_nbins, "-nbins", inputparser::optional);
    }


    // Read input track
    MovType movie;

    // Read input files
    for(int i = 0; i < ninfiles; ++i) {
        char* infilename = infiles[i];
        ifstream infile(infilename, ios::binary);
        if(!infile.is_open()) {
            cerr << "Open file failed: " << infilename << endl;
            return 1;
        }
        cout << "Load file " << infilename << endl;

        movie.read_merge(infile);

        infile.close();
    }

    if(movie.empty()) {
        cout << "No input read. Terminating." << endl;
        return 0;
    }

    // Extract pair dispersion for particles with adjacent indices
    if(!r0bin && !rprofile) {
        cout << "Write R2 to " << outfilename << endl;
        fstream R2writer(outfilename, ios::out);
        if(R2writer.is_open()) {
            R2writer << "# t\tnpoints\tR2_xyz\tR2_xy\tR2_z\n";
            extract_R2(movie, cube_length, dt, R2writer);
        } else {
            cout << "Open " << outfilename << " failed." << endl;
        }
        R2writer.close();
        cout << "Write R2 done" << endl;
    }

    // Extract pair dispersion for particles close to each other
    if(r0bin) {
        cout << "Extract bin ("<<r0bin_minmax.at(0)<<":"<<r0bin_minmax.at(1)<<") to " << outfilename << endl;
        fstream R2writer(outfilename, ios::out);
        if(R2writer.is_open()) {
            R2writer << "# t\tnpoints\tR2_xyz\tR2_xy\tR2_z\n";
            extract_bin(movie, cube_length, dt, r0bin_minmax.at(0), r0bin_minmax.at(1), R2writer);
        } else {
            cout << "Open " << outfilename << " failed." << endl;
        }
        R2writer.close();
        cout << "Write R2 done" << endl;

    }

    // Extract profile of pair dispersion at certain time index
    if(rprofile) {
        cout << "Extract profile at index " << rprofile_index << " to " << outfilename << endl;
        fstream R2writer(outfilename, ios::out);
        if(R2writer.is_open()) {
            R2writer << "# r\tdensity\n";
            extract_rprofile(movie, cube_length, dt, rprofile_index, rprofile_nbins, R2writer);
        } else {
            cout << "Open " << outfilename << " failed." << endl;
        }
        R2writer.close();
        cout << "Write R2 done" << endl;
    }

    return 0;
}

double pbc(double x0, double x1, double clx) {
    double dx = fabs(x1-x0);

    if(dx != dx || dx > 2*clx) {
        return -1.;
    }

    // First bring particles into one interval
    if(dx > clx) dx -= clx * floor(dx / clx);

    // Then see which direction is shorter
    if(dx > 0.5*clx) dx = clx - dx;

    return dx;
}

// Some error in the PBC routine messed up tracks
// that were passing the boundary.
// We remove them to get good results. flag false means remove.
void clean_track(MovType& movie, vector<double>& cube_length, vector<bool>& valid) {
    cout << "Clean" << endl;
    //cout << "!Attention: Strict Removal." << endl;
    double max_jump = 4;

    valid.resize(movie.get_NbTracks());

    for(unsigned long tr = 0; tr < movie.get_NbTracks(); tr++) {
        valid.at(tr) = true;

        for(unsigned long fr = 1; fr < movie.get_NbFrames(); ++fr) {
            double oldpos[3];
            double newpos[3];
            movie.getPos(fr-1, tr, oldpos[0], oldpos[1], oldpos[2]);
            movie.getPos(fr  , tr, newpos[0], newpos[1], newpos[2]);

            // If near boundary
            /*if(oldpos[0] < danger_zone || oldpos[0] > cube_length.at(0)-danger_zone || oldpos[1] < danger_zone || oldpos[1] > cube_length.at(1)-danger_zone) {
                valid.at(tr) = false;
                break;
            }*/

            double dx = pbc(oldpos[0], newpos[0], cube_length.at(0));
            double dy = pbc(oldpos[1], newpos[1], cube_length.at(1));
            // If unrealistic jump
            /*if(sqrt(dx*dx+dy*dy) >= max_jump) {
                cout << "Jump from " << oldpos[0] <<","<< oldpos[1] <<","<< oldpos[2] << "\n";
                cout << "to        " << newpos[0] <<","<< newpos[1] <<","<< newpos[2] << "\n";
            }*/
            if(!(dx >= 0 && dy >= 0 && sqrt(dx*dx+dy*dy) < max_jump)) {
                valid.at(tr) = false;
                break;
            }
        }
    }

    int ntracks = valid.size();
    for(size_t i = 0; i < valid.size(); ++i) {
        if(!valid.at(i)) --ntracks;
    }
    cout << "Clean Done. Valid: " << ntracks << "/" << valid.size() << endl;
}

void extract_R2(MovType& movie, vector<double>& cube_length, double dt, fstream& writer) {

    vector<bool> flag;
    clean_track(movie, cube_length, flag);

    for(unsigned long fr = 0; fr < movie.get_NbFrames(); ++fr) {
        double time = dt * movie.data.frames.at(fr).time;
        double R2 = 0.0;
        double R2xy = 0.0;
        double R2z = 0.0;
        int npoints = 0;

        for(unsigned long tr = 0; tr < movie.get_NbTracks(); tr += 2) {
            if(!flag.at(tr) || !flag.at(tr+1)) continue;

            double pos1[3];
            double pos2[3];
            movie.getPos(fr, tr, pos1[0], pos1[1], pos1[2]);
            movie.getPos(fr, tr+1, pos2[0], pos2[1], pos2[2]);

            double dx = pbc(pos1[0], pos2[0], cube_length.at(0));
            double dy = pbc(pos1[1], pos2[1], cube_length.at(1));
            double dz = pos1[2] - pos2[2];

            if(dx < 0 || dy < 0 || dz != dz) continue;

            double dx2 = dx*dx;
            double dy2 = dy*dy;
            double dz2 = dz*dz;

            R2 += dx2 + dy2 + dz2;
            R2xy += dx2 + dy2;
            R2z += dz2;
            npoints++;
        }

        if(npoints < 1) return;

        writer << time << '\t' << npoints << '\t' << (R2/npoints) << '\t' << (R2xy/npoints) << '\t' << (R2z/npoints) << '\n';
    }
}

void extract_bin(MovType& movie, vector<double>& cube_length, double dt, double mindiff, double maxdiff, fstream& writer){

    vector<bool> valid;
    clean_track(movie, cube_length, valid);

    vector<double> R2_list(movie.get_NbFrames(), 0.0);
    vector<double> R2xy_list(movie.get_NbFrames(), 0.0);
    vector<double> R2z_list(movie.get_NbFrames(), 0.0);
    vector<int> npoint_list(movie.get_NbFrames(), 0);

    for(unsigned long startframe = 0; startframe < movie.get_NbFrames(); ++startframe) {

        // All combinations of tracks at start frame
        for(unsigned long tr1 = 0; tr1 < movie.get_NbTracks(); tr1++) {
            if(!valid.at(tr1)) continue;

            for(unsigned long tr2 = tr1+1; tr2 < movie.get_NbTracks(); tr2++) {
                if(!valid.at(tr2)) continue;

                // Get difference between particle locations
                double pos1[3];
                double pos2[3];
                movie.getPos(startframe, tr1, pos1[0], pos1[1], pos1[2]);
                movie.getPos(startframe, tr2, pos2[0], pos2[1], pos2[2]);

                double dx = pbc(pos1[0], pos2[0], cube_length.at(0));
                double dy = pbc(pos1[1], pos2[1], cube_length.at(1));
                double dz = pos1[2] - pos2[2];

                // Skip invalid tracks
                if(dx < 0 || dy < 0 || dz != dz) continue;

                // Only use tracks in bin
                double r_sq = dx*dx+dy*dy+dz*dz;
                if(!(r_sq > mindiff*mindiff && r_sq < maxdiff*maxdiff)) continue;

                for(unsigned long frame = startframe; frame < movie.get_NbFrames(); ++frame) {

                    // Get difference between particle locations
                    double pos1[3];
                    double pos2[3];
                    movie.getPos(frame, tr1, pos1[0], pos1[1], pos1[2]);
                    movie.getPos(frame, tr2, pos2[0], pos2[1], pos2[2]);

                    double dx = pbc(pos1[0], pos2[0], cube_length.at(0));
                    double dy = pbc(pos1[1], pos2[1], cube_length.at(1));
                    double dz = pos1[2] - pos2[2];

                    // Skip invalid tracks
                    if(dx < 0 || dy < 0 || dz != dz) continue;

                    // Add R^2 to data
                    double r_sq = dx*dx+dy*dy+dz*dz;
                    R2_list.at(frame - startframe) += r_sq;
                    R2xy_list.at(frame - startframe) += dx*dx + dy*dy;
                    R2z_list.at(frame - startframe) += dz*dz;
                    npoint_list.at(frame - startframe)++;
                }
            }
        }
    }

    // We assume that times in movie are at equal distances
    dt *= (movie.data.frames.at(1).time - movie.data.frames.at(0).time);
    for(size_t i = 0; i < R2_list.size(); ++i) {
        int npoints = npoint_list.at(i);
        if(npoints < 1) continue;
        writer << i*dt << '\t' << npoints << '\t' << (R2_list.at(i)/npoints) << '\t' << (R2xy_list.at(i)/npoints) << '\t' << (R2z_list.at(i)/npoints) << '\n';
    }

}


void extract_rprofile(MovType& movie, vector<double>& cube_length, double dt, int rprofile_index, int nbins, fstream& writer){

    vector<bool> valid;
    clean_track(movie, cube_length, valid);

    vector<double> rlist;

    vector<int> zbins(10,0);

    // Get frame that corresponds to time index
    unsigned long int frame = 0;
    for(unsigned long fr = 0; fr < movie.get_NbFrames(); ++fr) {
        unsigned long int time = movie.data.frames.at(fr).time;
        if(time == (unsigned long int) rprofile_index) {
            frame = fr;
            break;
        }
    }

    // Get all distances at frame
    for(unsigned long tr = 0; tr < movie.get_NbTracks(); tr += 2) {
            if(!valid.at(tr) || !valid.at(tr+1)) continue;

            // Get difference between particle locations
            double pos1[3];
            double pos2[3];
            movie.getPos(frame, tr  , pos1[0], pos1[1], pos1[2]);
            movie.getPos(frame, tr+1, pos2[0], pos2[1], pos2[2]);

            double dx = pbc(pos1[0], pos2[0], cube_length.at(0));
            double dy = pbc(pos1[1], pos2[1], cube_length.at(1));
            double dz = pos1[2] - pos2[2];

            // Skip invalid tracks
            if(dx < 0 || dy < 0 || dz != dz) continue;

            // Only use tracks in bin
            double r = sqrt(dx*dx+dy*dy);//+dz*dz);
            rlist.push_back(r);

            // Bin z (additional feature)
            int zbin = 1 + (zbins.size()-2) * (pos1[2]+0.5);
            if(zbin < 1) zbins.at(0)++;
            else if(zbin > int(zbins.size()-2)) zbins.at(zbins.size()-1)++;
            else zbins.at(zbin)++;
            zbin = 1 + (zbins.size()-2) * (pos2[2]+0.5);
            if(zbin < 1) zbins.at(0)++;
            else if(zbin > int(zbins.size()-2)) zbins.at(zbins.size()-1)++;
            else zbins.at(zbin)++;
    }

    int npart = 0;
    for(size_t i = 0; i < zbins.size(); ++i) {
        npart += zbins.at(i);
    }
    cout << "At t = " << frame << " with "<<npart<<" parts:\n";
    for(size_t i = 0; i < zbins.size(); ++i) {
        cout << (double(100 * zbins.at(i)) / npart) << endl;
    }
    cout << endl;

    // Location of bins
    vector<double> bin_at(2,0.0);
    bin_at.at(1) = 1e-3;
    for(size_t i = 2; i < (size_t)nbins; ++i) {
        bin_at.push_back(1.17*bin_at.back());
    }

    // Bin the distances
    const int npoints = rlist.size();
    vector<double> rho(nbins,0.0);
    for(size_t i = 0; i < (size_t)npoints; ++i) {
        double r = rlist.at(i);
        for(size_t j = 0; j < bin_at.size()-1; ++j) {
            if(r<bin_at.at(j+1)) {
                double binwidth = bin_at.at(j+1) - bin_at.at(j);
                rho.at(j) += (1.0 / npoints) * (1.0 / binwidth);
                break;
            }
        }
    }

    // Write output
    for(size_t i = 0; i < bin_at.size()-1; ++i) {
        writer << (0.5*(bin_at.at(i)+bin_at.at(i+1))) << '\t' << rho.at(i) << '\n';
    }

}
/*int bin_x = xtobin(x, limits.at(0), limits.at(1), nbins_x);
int bin_y = xtobin(y, limits.at(3), limits.at(2), nbins_y);
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
bins.at(bin_x + nbins_x * bin_y)++;*/
int xtobin(double x, double xmin, double xmax, int nbins) {
    return ((x-xmin)*nbins) / (xmax-xmin);
}

double bintox(int bin, double xmin, double xmax, int nbins) {
    return xmin + (bin * (xmax-xmin))/nbins;
}




