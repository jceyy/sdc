#include "Mov.h"


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
    if(ID[0]=='T' && ID[1]=='R' && ID[2]=='K' && ID[3]=='M' && ID[4]=='G')
        return 1;
    //Type 2: TRACK
    if(ID[0]=='T' && ID[1]=='R' && ID[2]=='A' && ID[3]=='C' && ID[4]=='K')
        return 2;

    return 0;
}

/******************************************
* Specific TrackMov class                 *
******************************************/
//General Functions
TrackMov::TrackMov(){
    data.MegaTrack.clear();
    data.ntrks=0;
    data.fs=0;
    data.exposure=0;
    data.threshold=0;
    data.interpl_max=0;
    data.min_trk_len=0;
}

//Variable access
char*               TrackMov::get_ID()          { return data.ID; }
unsigned long int   TrackMov::get_fs()          { return data.fs; }
unsigned long int   TrackMov::get_exposure()    { return data.exposure; }
char                TrackMov::get_threshold()   { return data.threshold; }
char                TrackMov::get_interpl_max() { return data.interpl_max; }
unsigned short int  TrackMov::get_min_trk_len() { return data.min_trk_len; }
unsigned long int   TrackMov::get_NbTracks()    { return data.ntrks; }

//Calculated variables
unsigned long int TrackMov::get_NbFrames(){
    unsigned long int lastframe,NbFrames=0;
    for(unsigned long int i=0;i<data.ntrks;i++){
        tracks& t=data.MegaTrack.at(i);
        if(t.npoints>0){
            lastframe=data.MegaTrack.at(i).frame.at(t.npoints-1);
            if(lastframe>NbFrames) NbFrames=lastframe;
        }
    }
    return NbFrames;
}
unsigned int TrackMov::get_NbPart_at(int fr){
    unsigned int NbPart=0,ufr=fr;
    for(unsigned long int i=0;i<data.ntrks;i++){
        tracks& t=data.MegaTrack.at(i);
        if(t.npoints>0 && t.frame.at(0)<=ufr && t.frame.at(t.npoints-1)>=ufr){
            for(unsigned long int j=0;j<t.npoints;j++)
                if(t.frame.at(j)==ufr){
                    NbPart++; break;
                }
        }
    }
    return NbPart;
}
unsigned long int TrackMov::get_start(unsigned long int track){
    if(data.MegaTrack.at(track).npoints>0)
        return data.MegaTrack.at(track).frame.at(0);
    return 0;
}
unsigned long int TrackMov::get_end(unsigned long int track){
    unsigned long int lastframe=data.MegaTrack.at(track).frame.size()-1;
    if(data.MegaTrack.at(track).npoints>0)
        return data.MegaTrack.at(track).frame.at(lastframe);
    return 0;
}

//Specific Functions
void TrackMov::readmaxRanges(double& pRange,double& vRange,double& aRange){
    double r;
    pRange=0.; vRange=0.; aRange=0.;
    for(unsigned long int i=0;i<data.ntrks;i++){
        tracks& t=data.MegaTrack.at(i);
        for(unsigned long int j=0;j<t.npoints;j++){
            r=sqrt(t.x.at(j)*t.x.at(j)+t.y.at(j)*t.y.at(j)+t.z.at(j)*t.z.at(j));
            if(r>pRange) pRange=r;
        }
    }
}
bool TrackMov::getPos(unsigned long int fr,unsigned long int track,double& x,double& y,double& z){
    tracks& t=data.MegaTrack.at(track);
    unsigned long int part=t.npoints;
    if(t.npoints==0 || fr<t.frame.at(0) || fr>t.frame.at(t.npoints-1)) return false;
    for(unsigned long int i=0;i<t.npoints;i++){
        if(fr==t.frame.at(i)){
            part=i; break;
        }
    }
    if(part==t.npoints) return false;
    x=t.x.at(part);
    y=t.y.at(part);
    z=t.z.at(part);
    return true;
}
bool TrackMov::getVel(unsigned long int frame,unsigned long int track,double& x,double& y,double& z){
    x=y=z=0.;
    frame=track=0;
    return false;
}
bool TrackMov::getAcc(unsigned long int frame,unsigned long int track,double& x,double& y,double& z){
    x=y=z=0.;
    frame=track=0;
    return false;
}
bool TrackMov::getInterpl(unsigned long int fr,unsigned long int track,double& itpl){
    tracks& t=data.MegaTrack.at(track);
    unsigned long int part=t.npoints;
    if(t.npoints==0 || fr<t.frame.at(0) || fr>t.frame.at(t.npoints-1)) return false;
    for(unsigned long int i=0;i<t.npoints;i++){
        if(fr==t.frame.at(i)){
            part=i; break;
        }
    }
    if(part==t.npoints) return false;
    if(t.interpl.at(part)) itpl=1.;
    else itpl=0.;
    return true;
}
bool TrackMov::Read(std::ifstream& input){
    data.MegaTrack.clear();
    tracks localTrack;
    //variable in sizes used by binary file
    char Tchar;
    unsigned char Tuchar;
    short int Tsint;
    //unsigned long int Tulint;
    uint32_t Tulint;
    float Tfloat;


    ////////////////////// Header of the track file //////////////////////
    //fread(&TracksInThisFile.ID,5,1,f); //sizeof(char[5])
    for(int i=0;i<5;i++){
        input.read(&Tchar,sizeof(Tchar));
        data.ID[i]=Tchar;
    }
    data.ID[5]=0;
    //Error if file is not of "TRACK" type
    if(data.ID[0]!='T'||data.ID[1]!='R'||data.ID[2]!='A'||data.ID[3]!='C'||data.ID[4]!='K')
        return false;

    //fread(&NTRACKinF,4,1,f); //sizeof(unsigned long int)
    input.read((char*)&Tulint,sizeof(Tulint));
    data.ntrks=Tulint;
    //fread(&fs,4,1,f);//sizeof(unsigned long int)
    input.read((char*)&Tulint,sizeof(Tulint));
    data.fs=Tulint;
    //fread(&Exposure,4,1,f);//sizeof(unsigned long int)
    input.read((char*)&Tulint,sizeof(Tulint));
    data.exposure=Tulint;
    //fread(&threshold,1,1,f);//sizeof(unsigned char)
    input.read((char*)&Tuchar,sizeof(Tuchar));
    data.threshold=Tuchar;
    //fread(&interpl_max,1,1,f);//sizeof(unsigned char)
    input.read((char*)&Tuchar,sizeof(Tuchar));
    data.interpl_max=Tuchar;
    //fread(&min_trk_len,2,1,f);//sizeof(short int)
    input.read((char*)&Tsint,sizeof(Tsint));
    data.min_trk_len=Tsint;

    ////////////////////// Read Track by track  //////////////////////
    for(unsigned long int i=0;i<data.ntrks;i++){
        //fread(&NinTrack,4,1,f);//sizeof(unsigned long int)
        input.read((char*)&Tulint,sizeof(Tulint));
        localTrack.npoints=Tulint;
        for(unsigned int j=0;j<localTrack.npoints;j++){
            //fread(&frame,4,1,f);  //sizeof(unsigned long int)
            input.read((char*)&Tulint,sizeof(Tulint));
            localTrack.frame.push_back(Tulint);
            //fread(&x,4,1,f);  //sizeof(float)
            input.read((char*)&Tfloat,sizeof(Tfloat));
            localTrack.x.push_back(Tfloat);
            //fread(&y,4,1,f);  //sizeof(float)
            input.read((char*)&Tfloat,sizeof(Tfloat));
            localTrack.y.push_back(Tfloat);
            //fread(&z,4,1,f);  //sizeof(float)
            input.read((char*)&Tfloat,sizeof(Tfloat));
            localTrack.z.push_back(Tfloat);
            //fread(&sx,4,1,f);  //sizeof(float)
            input.read((char*)&Tfloat,sizeof(Tfloat));
            localTrack.sigmax.push_back(Tfloat);
            //fread(&sy,4,1,f);  //sizeof(float)
            input.read((char*)&Tfloat,sizeof(Tfloat));
            localTrack.sigmay.push_back(Tfloat);
            //fread(&I,1,1,f);  //sizeof(unsigned char)
            input.read((char*)&Tuchar,sizeof(Tuchar));
            localTrack.I.push_back(Tuchar);
            //fread(&interpl,1,1,f);  //sizeof(char)
            input.read((char*)&Tchar,sizeof(Tchar));
            localTrack.interpl.push_back(Tchar);
        }
        //FalseTrack(ThisTrack.npoints,ThisTrack);
        //dump_track_to_cout(ThisTrack);
        //ExtraToInterpolatedPointsCorr(ThisTrack);
        //dump_track_to_cout(ThisTrack);
        data.MegaTrack.push_back(localTrack);
        //Clear localTrack:
        localTrack.frame.clear();
        localTrack.x.clear();
        localTrack.y.clear();
        localTrack.z.clear();
        localTrack.sigmax.clear();
        localTrack.sigmay.clear();
        localTrack.I.clear();
        localTrack.interpl.clear();
    }
    return true;
}
