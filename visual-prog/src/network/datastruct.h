#ifndef DATASTRUCT_H
#define DATASTRUCT_H

#define UPACK_SIZE 2048
#define UDATA_SIZE (UPACK_SIZE/4-6)
#define UDESCR_LEN 32
#define UREQ_NEW 0xF619
#define UREQ_CNT 0x6A09
#define UTYPE_H 0x491C
#define UTYPE_P 0x08B1
#define UVERSION 1.1f

struct uPacket;

struct uHeader{
    unsigned short Type;
    unsigned short NumPackets;
    unsigned int Nt;
    unsigned int Nx;
    unsigned int Ny;
    uHeader() : Type(UTYPE_H), NumPackets(0), Nt(0), Nx(0), Ny(0) {}
};
//TYPE p
struct uSimHeader{
    float uVersion;    // Server code version
    float cVersion;    // GPU code version
    float tFinal; // Final time
    float dt;   // time step
    float Ra;   // Rayleigh-number
    float Pr;   // Prandtl-number
    float eta;  // penalization parameter
    float clx, cly, clz;   // cube lengths
    unsigned short nmx, nmy, nmz;   // mode numbers
    char device[UDESCR_LEN];        // device name
    char descr[UDESCR_LEN];  // description for simulation
    char Err1[UDESCR_LEN];   // Error string 1
    char Err2[UDESCR_LEN];   // Error string 2
    char Err3[UDESCR_LEN];   // Error string 3
    char Err4[UDESCR_LEN];   // Error string 4
};

// Payload is in [iStart, iEnd)
struct uSimPayload{
    int iStart;
    int iEnd;
    float Data[UDATA_SIZE];
};

struct uPacket{
    struct uHeader H;
    struct uSimPayload P;
};

typedef struct NetUserInput NetUserInput;
struct NetUserInput
{
    float radius; // modification radius
    float relPosX, relPosY; // where the mouse is
    int mode; // 1 = temperature, 2 = velocity,  0 (or anything else) = no modification
    float temperature;//, velX, velY; // temperature to set

};

struct uRequest{
    unsigned short Type;
    unsigned short resReq;
    unsigned int senderID;
    enum ImgType { None, Temperature, Particles, TempParticles } imgType;
    struct NetUserInput netUserInput;

    uRequest() : Type(UREQ_NEW), resReq(0) {}
};

#endif // DATASTRUCT_H
