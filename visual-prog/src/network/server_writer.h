#ifndef SERVER_WRITER_H
#define SERVER_WRITER_H

#include "../../../gpu_flow/code/cuda_defines.h"
#include "../../../gpu_flow/code/matrix/matrix_folder.h"
#include "../../../gpu_flow/code/matrix/matrix_host_iterator.h"
#include "../../../gpu_flow/code/operator/calculate_temperature_operator.h"
#include "../../../gpu_flow/code/particle/particle_tracer.h"
#include "serverclient.h"

/****************************************************************************
** This class manages the connection of the RBC GPU code to the UDP server
** in serverclient.h (which is part of RBCvisual).
****************************************************************************/
class server_writer : public RBCServer
{
public:
    server_writer(int Port);

    void setDeviceName(const char* str);
    void setCL(float clx, float cly, float clz);
    void setNM(unsigned int nmx, unsigned int nmy, unsigned int nmz);
    void setRaPrEta(float Ra, float Pr, float Eta);
    void setdttFinal(float dt, float tFinal);
    void setTemperature(matrix_folder *thetaFolder, calculate_temperature_operator* tempOpInit);
    void unsetTemperature();
    void setParticles(particle_tracer* tracer, float clx, float cly);
    void unsetParticles();

    int sendData(int Nt);

private:
    int Prepare(uPacket*& outData, unsigned short resReq, uRequest::ImgType imgType);

    bool _TemperatureSet, _ParticlesSet;
    uSimHeader _H;
    int _Nt;
    // Temperature tracing
    matrix_folder* _thetaFolder;
    calculate_temperature_operator* _tempOpInit;
    // Particle tracing
    particle_tracer* _tracer;
    float _clx, _cly;
};

#endif // SERVER_WRITER_H
