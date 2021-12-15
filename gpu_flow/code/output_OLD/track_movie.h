#ifndef TRACK_MOVIE_H
#define TRACK_MOVIE_H

#include <string>
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdint.h>
using namespace std;

/******************************************
* General Mov class                       *
******************************************/
class Mov {
    public:
    //Variable access
    virtual char* get_ID() =0;
    virtual unsigned long int get_fs() =0;
    virtual unsigned long int get_exposure() =0;
    virtual char get_threshold() =0;
    virtual char get_interpl_max() =0;
    virtual unsigned short int get_min_trk_len() =0;
    virtual unsigned long int get_NbTracks() =0;
    //Calculated variables
    virtual unsigned long int get_NbFrames() =0;
    virtual unsigned int get_NbPart_at(int frame) =0;
    virtual unsigned long int get_start(unsigned long int track)=0;
    virtual unsigned long int get_end(unsigned long int track)=0;
    //Specific Functions
    virtual void readmaxRanges(double& pRange,double& vRange,double& aRange) =0;
    virtual bool Read(std::ifstream& input)=0;
    virtual bool getPos(unsigned long int frame,unsigned long int track,double& x,double& y,double& z) =0;
    virtual bool getVel(unsigned long int frame,unsigned long int track,double& x,double& y,double& z) =0;
    virtual bool getAcc(unsigned long int frame,unsigned long int track,double& x,double& y,double& z) =0;
    virtual bool getInterpl(unsigned long int frame,unsigned long int track,double& itpl) =0;
    //General Functions
    static int getMovType(std::ifstream& input);
    virtual ~Mov(){}
};


/******************************************
* TrackMov structs                        *
******************************************/
#define TRACK_FLOAT_REAL float
struct track_movie_frame
{
    unsigned long int time;
    std::vector<TRACK_FLOAT_REAL> x; // positions of each point
    std::vector<TRACK_FLOAT_REAL> y;
    std::vector<TRACK_FLOAT_REAL> z;
    void clear() {
        x.clear();
        y.clear();
        z.clear();
    }
};

struct track_movie_data
{
    char ID[6]; // The ID code for a track file (usually "TRKSS")
    char version;
    unsigned long int num_tracks; // Nb of tracks in this file
    //unsigned long int num_frames; // Nb of frames in this file - frames.size()
    unsigned long int fs; // Framerate in frames per second
    std::vector<track_movie_frame> frames;
    std::string description;
};

/******************************************
* Specific track_movie class              *
******************************************/
class track_movie: public Mov {
    public:
    track_movie_data data;
    //General Functions
    track_movie();
    ~track_movie(){}
    //Variable access
    char* get_ID();
    unsigned long int get_fs();
    unsigned long int get_exposure();
    char get_threshold();
    char get_interpl_max();
    unsigned short int get_min_trk_len();
    unsigned long int get_NbTracks();
    //Calculated variables
    unsigned long int get_NbFrames();
    unsigned int get_NbPart_at(int /*frame*/);
    unsigned long int get_start(unsigned long int /*track*/);
    unsigned long int get_end(unsigned long int /*track*/);
    //Specific Functions
    void readmaxRanges(double& pRange,double& vRange,double& aRange);

    // Read input
    bool Read(std::ifstream& input);
    bool read_header(std::ifstream& input, unsigned long int& time_min, unsigned long int& time_max, unsigned long int& num_frames);
    bool read_append(std::ifstream& input);
    bool read_merge(std::ifstream& input);
    bool Write(std::ofstream& output);

    bool getPos(unsigned long int frame,unsigned long int track,double& x,double& y,double& z);
    bool getVel(unsigned long int frame,unsigned long int track,double& x,double& y,double& z);
    bool getAcc(unsigned long int frame,unsigned long int track,double& x,double& y,double& z);
    bool getInterpl(unsigned long int frame,unsigned long int track,double& itpl);
    void print();
    void clear();
    bool empty();
};



#endif // TRACK_MOVIE_H
