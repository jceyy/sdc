#include "server_writer.h"

server_writer::server_writer(int Port)
    : RBCServer(Port), _TemperatureSet(false), _ParticlesSet(false)
{
    // Initialize header
    memset(&_H, 0, sizeof(_H));
    _H.cVersion = 1.0;
    _H.uVersion = UVERSION;
}

void server_writer::setDeviceName(const char *str)
{
    strncpy(_H.device, str, UDESCR_LEN);
}

void server_writer::setCL(float clx, float cly, float clz)
{
    _H.clx = clx;
    _H.cly = cly;
    _H.clz = clz;
}

void server_writer::setNM(unsigned int nmx, unsigned int nmy, unsigned int nmz)
{
    _H.nmx = nmx;
    _H.nmy = nmy;
    _H.nmz = nmz;
}

void server_writer::setRaPrEta(float Ra, float Pr, float Eta)
{
    _H.Ra = Ra;
    _H.Pr = Pr;
    _H.eta = Eta;
}

void server_writer::setdttFinal(float dt, float tFinal)
{
    _H.dt = dt;
    _H.tFinal = tFinal;
}

void server_writer::setTemperature(matrix_folder *thetaFolder, calculate_temperature_operator *tempOpInit)
{
    _TemperatureSet = true;
    _tempOpInit = tempOpInit;
    _thetaFolder = thetaFolder;
}

void server_writer::unsetTemperature()
{
    _TemperatureSet = false;
}


void server_writer::setParticles(particle_tracer *tracer, float clx, float cly)
{
    _ParticlesSet = true;
    _tracer = tracer;
    _clx = clx;
    _cly = cly;
}

void server_writer::unsetParticles()
{
    _ParticlesSet = false;
}

int server_writer::sendData(int Nt, NetUserInput *data)
{
    // Set timestep index
    _Nt = Nt;

    // Send data
    return Send(data);
}

int server_writer::Prepare(uPacket *&outData, unsigned short resReq,
                           uRequest::ImgType imgType)
{
    // Preprocess data
    matrix_host_real *dataMatrix = 0;

    switch(imgType) {
    case uRequest::Temperature:
        // Write temperature field to data matrix
        if(!_TemperatureSet) return -1;
        {
            sprintf(_H.descr, "Temperature field at z = 0");

            vector<CUDA_FLOAT_REAL> at_z(1, 0.0);
            matrix_folder_real* TempFolder = _tempOpInit->calculate_operator_at(_thetaFolder, at_z);
            dataMatrix = new matrix_host_real(TempFolder->get_matrix(0), matrix_host_real::Copy);
            delete TempFolder;
        }
        break;
    case uRequest::TempParticles:
        if(!_TemperatureSet) return -1;
    case uRequest::Particles:
        if(!_ParticlesSet) return -1;
        {
            // Write particle positions to data matrix
            if(imgType == uRequest::TempParticles) {
                sprintf(_H.descr, "Temperature at z = 0 w/ Part");
            }else{
                sprintf(_H.descr, "Particles");
            }

            // Get particle positions from tracer
            matrix_folder_real* PartFolder = _tracer->get_particle_positions(particle_tracer::noClear);
            matrix_host_real* pos_x = new matrix_host_real(PartFolder->get_matrix(0), matrix_host_real::Copy);
            matrix_host_real* pos_y = new matrix_host_real(PartFolder->get_matrix(1), matrix_host_real::Copy);
            int num_part = pos_x->get_matrix_size(0);
            int index_t = pos_x->get_matrix_size(1) - 1;

            // Create matrix for output storage
            vector<int> indices_in(2), indices_out(3);

            if(imgType == uRequest::TempParticles) {
                vector<CUDA_FLOAT_REAL> at_z(1, 0.0);
                matrix_folder_real* TempFolder = _tempOpInit->calculate_operator_at(_thetaFolder, at_z);
                dataMatrix = new matrix_host_real(TempFolder->get_matrix(0), matrix_host_real::Copy);
                delete TempFolder;
            } else {
                indices_out[0] = (resReq <= 0) ? 128 : resReq;
                indices_out[1] = indices_out[0];
                indices_out[2] = 1;
                dataMatrix = new matrix_host_real(indices_out);
                dataMatrix->init_zeros();
            }

            // Plot particles into output image
            int xSize = dataMatrix->get_matrix_size(0);
            int ySize = dataMatrix->get_matrix_size(1);
            CUDA_FLOAT_REAL entry;

            // Reverse order so that particle 0 is always visible
            for(int i = num_part - 1; i >= 0; --i) {

                // Get particle position
                indices_in[0] = i;
                indices_in[1] = index_t;
                CUDA_FLOAT_REAL x = pos_x->get_entry(indices_in);
                CUDA_FLOAT_REAL y = pos_y->get_entry(indices_in);

                CUDA_FLOAT_REAL x_grid = (xSize * x) / _clx;
                CUDA_FLOAT_REAL y_grid = (ySize * y) / _cly;
                indices_out[0] = x_grid - floor(x_grid / xSize) * xSize;
                indices_out[1] = y_grid - floor(y_grid / ySize) * ySize;
                indices_out[2] = 0;

                // Check index out of bounds
                if(indices_out[0] < 0 || indices_out[0] >= xSize
                        || indices_out[1] < 0 || indices_out[1] >= ySize) {
                    setError("Particle %d at (%f,%f) out of bounds.", i, x, y);
                    continue;
                }

                // Set z-index and color
                if(imgType == uRequest::TempParticles) {
                    entry = 0;
                } else {
                    entry = dataMatrix->get_entry(indices_out) + 1;
                }

                dataMatrix->set_entry(indices_out, entry);
            }

            delete PartFolder;
            delete pos_x;
            delete pos_y;
        }
        break;
    default:
        setError("Incorrect imgType");
        return -1;
    }

    // Set resolution of output image
    int Nx = resReq, Ny = resReq;
    if(Nx <= 0 || Nx > dataMatrix->get_matrix_size(0)) Nx = dataMatrix->get_matrix_size(0);
    if(Ny <= 0 || Ny > dataMatrix->get_matrix_size(1)) Ny = dataMatrix->get_matrix_size(1);

    // Number of packets: 1 header, roundUp(Nx*Ny/UDATA_SIZE) data
    int NumPack = 1 + (Nx*Ny + UDATA_SIZE - 1)/UDATA_SIZE;
    outData = new uPacket[NumPack];
    for(int i = 0; i < NumPack; i++) {
        outData[i].H.NumPackets = NumPack;
        outData[i].H.Nt = _Nt;
        outData[i].H.Nx = Nx;
        outData[i].H.Ny = Ny;
    }

    // Add errors into header
    // TODO: Have ErrorHandling-class with 4 last error strings
    strncpy(_H.Err1, ErrMsg(), UDESCR_LEN);

    // Make header packet
    outData[0].H.Type = UTYPE_H;
    uSimHeader* SimH = (uSimHeader*) &(outData[0].P);
    *SimH = _H;

    // Make data packets
    switch(imgType) {
    case uRequest::Temperature:
    case uRequest::Particles:
    case uRequest::TempParticles:
    {
        vector<int> indices(3, 0);
        indices[2] = dataMatrix->get_matrix_size(2) / 2;

        // Build packets
        int iTot = 0;
        int iPack = 1;
        int iInPack = 0;
        uSimPayload* SimP = &(outData[iPack].P);
        SimP->iStart = iTot;
        for (int iX = 0; iX < Nx; iX++) {
            for (int iY = 0; iY < Ny; iY++,iInPack++,iTot++) {

                if (iInPack >= UDATA_SIZE) {
                    // Open new Pack
                    SimP->iEnd = iTot;
                    iPack++;
                    SimP = &(outData[iPack].P);
                    SimP->iStart = iTot;
                    iInPack = 0;
                }

                indices[0] = iX * dataMatrix->get_matrix_size(0) / Nx;
                indices[1] = iY * dataMatrix->get_matrix_size(1) / Ny;
                SimP->Data[iInPack] = dataMatrix->get_entry(indices);
            }
        }
        SimP->iEnd = iTot;
        for (iPack++; iPack < NumPack; iPack++) {
            outData[iPack].P.iStart = 0;
            outData[iPack].P.iEnd = 0;

        }
    }
    break;
    default:
        ;
    }

    // Clear
    delete dataMatrix;

    return 0;
}
