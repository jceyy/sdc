/****************************************************************************
** This file contains the server and client code for managing a UDP
** connection.
**
** The RBCServer and RBCClient classes must be reimplemented to provide
** the virtual functions
** int RBCServer::Prepare(uPacket*& outData, void* inData)
** and
** int RBCClient::Add(uPacket*& inData, void* outData);
**
** The Prepare-function must reserve memory for the outData packets of the
** size that is stored in outData[0].NumPackets, and fill these packets.
** If no memory is reserved, outData must be set to 0. The function returns
** 0 if successful and -1 if unsuccessful.
****************************************************************************/
#ifndef SERVERCLIENT_H
#define SERVERCLIENT_H

#include <netinet/in.h>
#include "datastruct.h"
using namespace std;

#define ERRSTR_LEN 128
#define HOSTNAME_LEN 128
#define UDPBUF_SIZE (1024*1024*3)
#define MAX_CONN 5

/******************************************
* Requirements for PrepFn: allocates      *
* memory for outData (0 if none allocated)*
* and stores size in PacketNum of each    *
* uPacket. At least 1 packet must be      *
* prepared.                               *
******************************************/

class RBCServerClient
{
public:
    RBCServerClient(unsigned int Port, const char* HostName);

    char* HostName();
    int Port();
    bool Connected();

    void Disconnect();

    bool hasError();
    void resetError();
    char *ErrMsg();

protected:
    bool _Connected;
    unsigned int _Port;
    int _SockListen;
    struct sockaddr_in _ServerAddr;
    char _HostName[HOSTNAME_LEN];

    //void setError(const char* Msg);
    void setError(const char* Msg, ...);

private:

    bool _Error;
    char _ErrString[ERRSTR_LEN];
    char _NoErrString[1];
};

class RBCServer : public RBCServerClient
{
public:
    RBCServer(unsigned int Port);
    ~RBCServer();
    void Clear();

protected:
    int Send(net_dataSend *data);
	int Send();
    virtual int Prepare(uPacket*& outData, unsigned short resReq, uRequest::ImgType imgType) = 0;

private:
    int SendToConnection(int iConn, sockaddr_in *OutAddr, uRequest &Req);

    uPacket* _outData[MAX_CONN];
    int _iOutData[MAX_CONN];

    unsigned int _Connections[MAX_CONN];
    bool _ConnDone[MAX_CONN];
    int _nConnections;
};

class RBCClient : public RBCServerClient
{
private:
    unsigned int _senderID;

protected:
    virtual int Add(uPacket& inData, void* outData) = 0;

public:
    enum ReqType { noReq, newReq, cntReq } ;
    RBCClient(unsigned int Port, const char* HostName);
    ~RBCClient();

    int Receive(void *outData,  ReqType reqType, uRequest::ImgType imgType, unsigned short resReq); //int==>ReqType

};

#endif // SERVERCLIENT_H


