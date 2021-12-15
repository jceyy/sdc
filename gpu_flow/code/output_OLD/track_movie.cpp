#include "track_movie.h"


/******************************************
* General Mov class                       *
******************************************/
int Mov::getMovType(std::ifstream& input){
    //Error if no file is open
    if(!input.is_open())
        return 0;

    char ID[6];
    for(int i=0;i<5;i++)
        input.read((char *)&ID[i],sizeof(char));
    ID[5]=0;

    //Return get pointer to beginning of file
    input.seekg(0, std::ios::beg);

    //Type 1: TRKMG
    if(strncmp(ID,"TRKMG",5)==0)
        return 1;
    //Type 2: TRACK
    if(strncmp(ID,"TRACK",5)==0)
        return 2;
    //Type 3: TRKSS
    if(strncmp(ID,"TRKSS",5)==0)
        return 3;

    return 0;
}

/******************************************
* Specific track_movie class                 *
******************************************/
//General Functions
track_movie::track_movie(){
    strncpy(data.ID, "TRKSS", 6);
    data.version = 'a';
    data.frames.clear();
    data.num_tracks = 0;
}

//Variable access
char*               track_movie::get_ID()          { return data.ID; }
unsigned long int   track_movie::get_fs()          { return data.fs; }
unsigned long int   track_movie::get_exposure()    { return 0; }
char                track_movie::get_threshold()   { return 0; }
char                track_movie::get_interpl_max() { return 0; }
unsigned short int  track_movie::get_min_trk_len() { return data.frames.size(); }
unsigned long int   track_movie::get_NbTracks()    { return data.num_tracks; }

//Calculated variables
unsigned long int track_movie::get_NbFrames(){
    return data.frames.size();
}
unsigned int track_movie::get_NbPart_at(int /*fr*/){
    return data.num_tracks;
}
unsigned long int track_movie::get_start(unsigned long int /*track*/){
    if(data.frames.size()>0)
        return data.frames.front().time;
    return 0;
}
unsigned long int track_movie::get_end(unsigned long int /*track*/){
    if(data.frames.size()>0)
        return data.frames.back().time;
    return 0;
}

//Specific Functions
void track_movie::readmaxRanges(double& pRange,double& vRange,double& aRange) {
    double r;
    pRange = 0.;
    vRange = 0.;
    aRange = 0.;
    for(unsigned long int fr = 0; fr < data.frames.size(); ++fr) {
        track_movie_frame& frame = data.frames.at(fr);
        for(unsigned long int p = 0; p < data.num_tracks; ++p) {
            double x = frame.x.at(p);
            double y = frame.y.at(p);
            double z = frame.z.at(p);
            r = sqrt(x*x + y*y + z*z);
            if(r > pRange) pRange = r;
        }
    }
}
bool track_movie::getPos(unsigned long int fr, unsigned long int track, double& x, double& y, double& z) {
    track_movie_frame& frame = data.frames.at(fr);
    x = frame.x.at(track);
    y = frame.y.at(track);
    z = frame.z.at(track);
    return true;
}
bool track_movie::getVel(unsigned long int frame,unsigned long int track,double& x,double& y,double& z){
    x=y=z=0.;
    frame=track=0;
    return false;
}
bool track_movie::getAcc(unsigned long int frame,unsigned long int track,double& x,double& y,double& z){
    x=y=z=0.;
    frame=track=0;
    return false;
}
bool track_movie::getInterpl(unsigned long int /*fr*/,unsigned long int /*track*/,double& itpl){
    itpl = 0.;
    return false;
}

bool track_movie::Read(std::ifstream& input){

    int type = getMovType(input);
    if(type != 3) {
        cout << "Input of type " << type << " not compatible with TRKSS (type 3) reader." << endl;
        return false;
    }

    unsigned long int time_min;
    unsigned long int time_max;
    unsigned long int num_frames;

    if(!read_header(input, time_min, time_max, num_frames))
        return false;

    track_movie_frame frame;

    //variable in sizes used by binary file
    char Tchar;
    //unsigned long int Tulint;
    uint32_t Tulint;
    TRACK_FLOAT_REAL Tfloat;

    ////////////////////// Read frame by frame  //////////////////////
    for(unsigned long int fr = 0; fr < num_frames; ++fr) {

        // unsigned long int time
        input.read((char*)&Tulint,sizeof(Tulint));
        frame.time = Tulint;

        for(unsigned int j = 0; j < data.num_tracks; j++) {

            // TRACK_FLOAT_REAL x
            input.read((char*)&Tfloat,sizeof(Tfloat));
            frame.x.push_back(Tfloat);

            // TRACK_FLOAT_REAL y
            input.read((char*)&Tfloat,sizeof(Tfloat));
            frame.y.push_back(Tfloat);

            // TRACK_FLOAT_REAL z
            input.read((char*)&Tfloat,sizeof(Tfloat));
            frame.z.push_back(Tfloat);
        }

        // Add frame to movie
        data.frames.push_back(frame);

        // Clear frame:
        frame.clear();
    }
    return true;
}

bool track_movie::read_header(std::ifstream& input, unsigned long int& time_min, unsigned long int& time_max, unsigned long int& num_frames) {

    //variable in sizes used by binary file
    char Tchar;
    //unsigned long int Tulint;
    uint32_t Tulint;
    TRACK_FLOAT_REAL Tfloat;


    ////////////////////// Header of the track file //////////////////////
    for(int i = 0; i < 5; ++i) {
        input.read(&Tchar, sizeof(Tchar));
        if(Tchar != data.ID[i])
            return false;
    }
    data.ID[5] = 0;

    // char version
    input.read(&Tchar,sizeof(Tchar));
    if(Tchar != data.version)
        return false;

    // unsigned long int num_tracks
    input.read((char*)&Tulint,sizeof(Tulint));
    data.num_tracks = Tulint;

    // unsigned long int fs (framerate in fps)
    input.read((char*)&Tulint,sizeof(Tulint));
    data.fs = Tulint;

    // string description
    input.read((char*)&Tulint,sizeof(Tulint)); // Length
    data.description.resize(Tulint);
    input.read(&(data.description[0]), Tulint);

    ////////////////////// Additional information /////////////////////
    // unsigned long int time_min
    input.read((char*)&Tulint,sizeof(Tulint));
    time_min = Tulint;

    // unsigned long int time_max
    input.read((char*)&Tulint,sizeof(Tulint));
    time_max = Tulint;

    // unsigned long int num_frames
    input.read((char*)&Tulint,sizeof(Tulint));
    num_frames = Tulint;

    return true;
}

bool track_movie::read_append(std::ifstream &input)
{
    // Appending to empty movie is reading.
    if(empty()) return Read(input);

    // Read movie
    track_movie movie;
    if(!movie.Read(input)) return false;

    // Check if size fits
    if(data.num_tracks != movie.data.num_tracks) return false;

    // Append frames
    for(unsigned int fr = 0; fr < movie.data.frames.size(); ++fr) {
        data.frames.push_back(movie.data.frames.at(fr));
    }
}

bool track_movie::read_merge(std::ifstream &input)
{
    // Appending to empty movie is reading.
    if(empty()) return Read(input);

    // Read movie
    track_movie movie;
    if(!movie.Read(input)) return false;

    // Check if size fits
    if(data.frames.size() != movie.data.frames.size()) return false;

    // Append tracks
    data.num_tracks += movie.data.num_tracks;
    for(unsigned int fr = 0; fr < data.frames.size(); ++fr) {
        for(unsigned int tr = 0; tr < movie.data.num_tracks; ++tr) {
            data.frames.at(fr).x.push_back(movie.data.frames.at(fr).x.at(tr));
            data.frames.at(fr).y.push_back(movie.data.frames.at(fr).y.at(tr));
            data.frames.at(fr).z.push_back(movie.data.frames.at(fr).z.at(tr));
        }
    }
}

bool track_movie::Write(std::ofstream& output){
    //print();
    //variable in sizes used by binary file
    char Tchar;
    //unsigned long int Tulint;
    uint32_t Tulint;
    TRACK_FLOAT_REAL Tfloat;


    ////////////////////// Header of the track file //////////////////////
    for(int i = 0; i < 5; ++i) {
        Tchar = data.ID[i];
        output.write(&Tchar, sizeof(Tchar));
    }

    // char version
    Tchar = data.version;
    output.write(&Tchar,sizeof(Tchar));

    // unsigned long int num_tracks
    Tulint = data.num_tracks;
    output.write((char*)&Tulint,sizeof(Tulint));

    // unsigned long int fs (framerate in fps)
    Tulint = data.fs;
    output.write((char*)&Tulint,sizeof(Tulint));

    // string description
    Tulint = data.description.length();
    output.write((char*)&Tulint,sizeof(Tulint)); // Length
    output.write(&(data.description[0]), data.description.length());

    ////////////////////// Additional information /////////////////////
    // unsigned long int time_min
    Tulint = get_start(0);
    output.write((char*)&Tulint,sizeof(Tulint));

    // unsigned long int time_max
    Tulint = get_end(0);
    output.write((char*)&Tulint,sizeof(Tulint));

    // unsigned long int num_frames
    Tulint = data.frames.size();
    output.write((char*)&Tulint,sizeof(Tulint));


    ////////////////////// Read frame by frame  //////////////////////
    for(unsigned long int fr = 0; fr < data.frames.size(); ++fr) {
        track_movie_frame& frame = data.frames.at(fr);

        // unsigned long int time
        Tulint = frame.time;
        output.write((char*)&Tulint,sizeof(Tulint));

        for(unsigned int j = 0; j < data.num_tracks; j++) {

            // TRACK_FLOAT_REAL x
            Tfloat = frame.x.at(j);
            output.write((char*)&Tfloat,sizeof(Tfloat));

            // TRACK_FLOAT_REAL y
            Tfloat = frame.y.at(j);
            output.write((char*)&Tfloat,sizeof(Tfloat));

            // TRACK_FLOAT_REAL z
            Tfloat = frame.z.at(j);
            output.write((char*)&Tfloat,sizeof(Tfloat));
        }
    }
    return true;
}

void track_movie::print()
{
    cout << "track_movie " << data.ID << " v " << data.version << endl;
    cout << data.num_tracks << " tracks of " << data.frames.size() << " each." << endl;
    cout << "Description: \"" << data.description << "\"" << endl;
    for(unsigned int fr = 0; fr < data.frames.size(); ++fr) {
        if(fr >= data.frames.size()) {
            cout << "Frame "<<fr<<" >= "<<data.frames.size() << endl;
            continue;
        }
        cout << "Frame "<<fr<<" at time "<<data.frames.at(fr).time <<endl;
        for(unsigned int p = 0; p < data.num_tracks; ++p) {
            cout << "[" <<data.frames.at(fr).x.at(p) << "," << data.frames.at(fr).y.at(p) << "," << data.frames.at(fr).z.at(p) << "], ";
            if(p > 20) {
                cout << "...";
                break;
            }
        }
        cout << endl;
    }
}

void track_movie::clear() {
    data.frames.clear();
}

bool track_movie::empty() {
    return (data.frames.size() == 0);
}
