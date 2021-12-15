#ifndef MOV_H
#define MOV_H

#include <string>
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
struct tracks
{
    unsigned long int npoints;                // # of points in this track
    std::vector<unsigned long int> frame;          // frame number of each points (time)
    std::vector<double> x;           // positions and width of gaussian fit of each point
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> sigmax;
    std::vector<double> sigmay;
    std::vector<unsigned char> I;                  // Intensity of the peak (average)
    std::vector<char> interpl;                     // =1 means this is an interpolated point
};

struct TracksInFile
{
    std::vector<tracks> MegaTrack;
    std::string filename;
    char ID[6]; // The ID code for a track file (usually "TRACK")
    unsigned long int ntrks; // Nb of tracks in this file
    unsigned long int fs; // Frame rate in frame per second
    unsigned long int exposure; // exposure in microsecond
    unsigned char threshold; // average of the thresholds used
    unsigned char interpl_max; // max # of interpolation points
    unsigned short int min_trk_len; // min track length to save
};
/******************************************
* Specific TrackMov class                 *
******************************************/
class TrackMov: public Mov {
    public:
    TracksInFile data;
    //General Functions
    TrackMov();
    ~TrackMov(){}
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
    unsigned int get_NbPart_at(int frame);
    unsigned long int get_start(unsigned long int track);
    unsigned long int get_end(unsigned long int track);
    //Specific Functions
    void readmaxRanges(double& pRange,double& vRange,double& aRange);
    bool Read(std::ifstream& input);
    bool getPos(unsigned long int frame,unsigned long int track,double& x,double& y,double& z);
    bool getVel(unsigned long int frame,unsigned long int track,double& x,double& y,double& z);
    bool getAcc(unsigned long int frame,unsigned long int track,double& x,double& y,double& z);
    bool getInterpl(unsigned long int frame,unsigned long int track,double& itpl);
};



#endif // MOV_H
