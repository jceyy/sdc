#include "datafile.h"
#include <stdarg.h>
#include <stdio.h>

DataFile::DataFile()
{
    _HasHeader = false;

    _Nx = 0;
    _Ny = 0;
    _Nt = 0;

    _Min = FLT_MAX;
    _Max = -FLT_MAX;

    _DataSize = 0;
    _Data = 0;

    _CountPackets = 0;
    _NumPackets = -1;
}

DataFile::~DataFile() {
    if(_Data) delete [] _Data;
}

unsigned int DataFile::Nx() {
    return _Nx;
}

unsigned int DataFile::Ny() {
    return _Ny;
}

unsigned int DataFile::Size() {
    return _Nx*_Ny;
}

unsigned int DataFile::Nt() {
    return _Nt;
}

float* DataFile::Data() {
    return _Data;
}

float DataFile::at(unsigned int ix, unsigned int iy)
{
    if(ix < _Nx && iy < _Ny) {
        return _Data[ix*_Ny + iy];
    }
    return 0;
}

float DataFile::RelAt(unsigned int ix, unsigned int iy)
{
    if(_Max < _Min) return 0;
    float ret = (_Min+at(ix,iy))/(_Max-_Min);
    if(ret < 0) return 0;
    if(ret > 1) return 1;
    return ret;
}

int DataFile::AddPacket(uPacket& P) {

    if(_CountPackets == 0) {
        _Nx = P.H.Nx;
        _Ny = P.H.Ny;
        _Nt = P.H.Nt;
        _NumPackets = P.H.NumPackets;

        if(_Nx*_Ny > 2000000) {
            // Error in packet
            setError("Error in Packet: (_Nx,_Ny) = %i,%i\n", _Nx, _Ny);
            return -1;
        }
        if(_DataSize != _Nx*_Ny) {
            delete [] _Data;
            _Data = new float[_DataSize = _Nx*_Ny];
        }

        _Min = FLT_MAX;
        _Max = -FLT_MAX;
        for(unsigned int i = 0; i < _DataSize; i++) _Data[i] = 0;
    }

    if(P.H.Type == UTYPE_H) {
        // Decompose simulation header packet

        uSimHeader& H = *((uSimHeader*) &(P.P));
        _HasHeader = true;
        _H = H;

        //setError("At t = %i: Add Header\n", _Nt);
    } else if(P.H.Type == UTYPE_P) {
        // Decompose data payload packet

        uSimPayload& Data = P.P;
        if(Data.iStart < 0 || (unsigned int)Data.iEnd > _DataSize) {
            setError("Error in Packet: data index = (%i,%i)\n", Data.iStart, Data.iEnd);
            return -1;
        }
        for(int i = Data.iStart; i < Data.iEnd; ++i) {
            float D = Data.Data[i-Data.iStart];
            if(D != D) D = 0;
            else if(D != 0) {
                if(D > _Max) _Max = D;
                if(D < _Min) _Min = D;
            }
            _Data[i] = D;
        }
        //setError("At t = %i: Add Packet [%i:%i)\n", _Nt, Data.iStart, Data.iEnd);

    } else {
        // Error in packet
        setError("Error in Packet: Type = %hi\n", P.H.Type);
        return -1;
    }
	_CountPackets=(_CountPackets+1)%90;
    //_CountPackets = (_CountPackets+1)%319; //_CountPackets++;
    return 0;
}

void DataFile::Copy(const DataFile& Original) {

    _Nx = Original._Nx;
    _Ny = Original._Ny;
    _Nt = Original._Nt;

    if(_DataSize != _Nx*_Ny) {
        delete [] _Data;
        _Data = new float[_DataSize = _Nx*_Ny];
    }
    for(unsigned int i = 0; i < _DataSize; i++) {
        _Data[i] = Original._Data[i];
    }

    // Header information
    _HasHeader = Original._HasHeader;
    _H = Original._H;

    // Data information
    _Min = Original._Min;
    _Max = Original._Max;

    _NumPackets = Original._NumPackets;
    _CountPackets = Original._CountPackets;
}

bool DataFile::isComplete()
{
    return (_CountPackets == _NumPackets);
}

bool DataFile::hasHeader()
{
    return _HasHeader;
}

float DataFile::Completeness()
{
    return double(_CountPackets)/double(_NumPackets);
}

void DataFile::setError(const char *Msg, ...)
{
    va_list Args;
    va_start(Args, Msg);
    vfprintf(stderr, Msg, Args);
    va_end(Args);
}



/******************************************
* Data folder                             *
******************************************/
DataFolder::DataFolder()
{
    //_LastAdd.start();
    _NFiles = 0;
    _hasChanged = false;
}

DataFolder::~DataFolder()
{
    for(unsigned int i = 0; i < _NFiles; i++) {
        delete _Files[i];
    }
}

int DataFolder::Add(uPacket &inData)
{
    DataFile* D = 0;

    for(unsigned int i = 0; i < _NFiles; i++) {

        if(_Files[i]->Nt() == (unsigned int)inData.H.Nt) {
            D = _Files[i];
            break;
        }
    }

    if(D == 0) {
        // If not added: Create new file
        if(_NFiles >= FOLDER_NFILES) {
            delete _Files[0];
            for(unsigned int i = 1; i < _NFiles; i++) {
                _Files[i-1] = _Files[i];
            }
        } else {
            _NFiles++;
        }
        D = _Files[_NFiles-1] = new DataFile;
    }

    _hasChanged = true;
//    _LastAdd.start();

    return D->AddPacket(inData);
}

DataFile *DataFolder::GetBest()
{
    double Cmax = 0;
    DataFile* Dmax = 0;
    for(int i = _NFiles - 1; i >= 0; i--) {

        DataFile* D = _Files[i];
        if(D->isComplete()) return D;

        double C = D->Completeness();
        if(C > Cmax) {
            Cmax = C;
            Dmax = D;
        }
    }

    return Dmax;
}

DataFile *DataFolder::GetNewest()
{
    if(_NFiles > 0) return _Files[_NFiles - 1];
    return 0;
}

bool DataFolder::hasChanged()
{
    return _hasChanged;
}

bool DataFolder::hasChangedReset()
{
    bool ret = _hasChanged;
    _hasChanged = false;
    return ret;
}

