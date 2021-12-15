#ifndef DATAFILE_H
#define DATAFILE_H


#include <math.h>
#include <iostream>
#include <float.h>
//#include "/home/ecps/Desktop/Finished project/Code visualisation/network/serverclient.h"
//#include "/home/ecps/Desktop/Finished project/Code visualisation/network/datastruct.h"
#include "serverclient.h"
#include "datastruct.h"

#define FOLDER_NFILES 5

#include <iostream>
using namespace std;

class DataFile
{
public:
    DataFile();
    ~DataFile();

    unsigned int Nx();
    unsigned int Ny();
    unsigned int Size();
    unsigned int Nt();
    float *Data();
    float at(unsigned int ix, unsigned int iy);
    float RelAt(unsigned int ix, unsigned int iy);

    int AddPacket(uPacket& P);
    void Copy(const DataFile& Original);
    bool isComplete();
    bool hasHeader();
    float Completeness();
    short NumPackets() { return _NumPackets; }
    short CountPackets() { return _CountPackets; }
    float Min() { return _Min; }
    float Max() { return _Max; }
    const uSimHeader& H() { return _H; }

private:
    void setError(const char* Msg, ...);

    unsigned int _Nx;
    unsigned int _Ny;
    unsigned int _Nt;
    unsigned int _DataSize;

    // Header information
    bool _HasHeader;
    uSimHeader _H;

    // Data information
    float* _Data;
    float _Min, _Max;

    short _NumPackets, _CountPackets;
};

class DataFolder
{
public:
    DataFolder();
    ~DataFolder();

    int Add(uPacket& inData);
    DataFile* GetBest();
    DataFile* GetNewest();
    bool hasChanged();
    bool hasChangedReset();

private:
    DataFile* _Files[FOLDER_NFILES];
    unsigned int _NFiles;
    bool _hasChanged;

};

class DataFolderClient : public RBCClient
{
public:
    DataFolderClient(unsigned int Port, const char* HostName)
        : RBCClient(Port, HostName) {}
private:
    int Add(uPacket& inData, void* outData) {
        return ((DataFolder*) outData)->Add(inData);
    }
};

#endif // DATAFILE_H
