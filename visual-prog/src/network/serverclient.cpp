#include "serverclient.h"

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>
#include <iostream>
#ifndef DBGOUT
#define DBGOUT(x) { fprintf(stderr, "Debug out at %s:%d: %s\n",__FILE__ ,__LINE__, x);}
#endif

RBCServerClient::RBCServerClient(unsigned int Port, const char* HostName)
    : _Connected(false), _Port(Port)
{
    // Set host name
    strncpy(_HostName, HostName, HOSTNAME_LEN);

    // Initialize error handling
    _Error = false;
    _ErrString[0] = 0;
    _NoErrString[0] = 0;
}

char *RBCServerClient::HostName()
{
    return _HostName;
}

int RBCServerClient::Port()
{
    return _Port;
}

bool RBCServerClient::Connected()
{
    return _Connected;
}

void RBCServerClient::Disconnect()
{
    _Connected = false;
}

/*void RBCServerClient::setError(const char *Msg) {
    strncpy(_ErrString, Msg, ERRSTR_LEN);
    _Error = true;
}*/

void RBCServerClient::setError(const char *Msg, ...)
{
    _Error = true;

    va_list Args;
    va_start(Args, Msg);
    vsprintf(_ErrString, Msg, Args);
    va_end(Args);

    fprintf(stderr, "%s\n", _ErrString);
}

bool RBCServerClient::hasError() {
    return _Error;
}

void RBCServerClient::resetError()
{
    _Error = false;
    _ErrString[0] = 0;
}

char* RBCServerClient::ErrMsg() {
    if(_Error) return _ErrString;
    return _NoErrString;
}


/******************************************
* RBC Server                              *
******************************************/
RBCServer::RBCServer(unsigned int Port)
    : RBCServerClient(Port, "localhost")
{
    // Initialize data fields
    _nConnections = 0;
    for(int iConn = 0; iConn < MAX_CONN; ++iConn) {
        _outData[iConn] = 0;
    }

    // Create socket
    _SockListen = socket(AF_INET, SOCK_DGRAM, 0);
    if (_SockListen < 0) {
        setError("Error opening socket (in RBCServer::RBCServer)");
        return;
    }

    // Set flag for non-blocking socket
    fcntl(_SockListen, F_SETFL, fcntl(_SockListen, F_GETFL) | O_NONBLOCK);

    // Bind socket to server
    int AddrLen = sizeof(_ServerAddr);
    memset(&_ServerAddr, 0, AddrLen);
    _ServerAddr.sin_family = AF_INET;
    _ServerAddr.sin_addr.s_addr = INADDR_ANY;
    _ServerAddr.sin_port = htons(_Port);
    if (bind(_SockListen, (struct sockaddr *) &_ServerAddr, AddrLen) < 0) {
        setError("Error on binding (in RBCServer::RBCServer)");
        return;
    }

    // Connection established
    _Connected = true;
    //cout << "Connected on sock "<<_SockListen<<", Port "<<_Port<<", Server "<<_HostName<<endl;
}

RBCServer::~RBCServer()
{
    Clear();
    close(_SockListen);
}

int RBCServer::Send() {

    if(!_Connected) {
        return 0;
    }
    for(int i = 0; i < _nConnections; ++i) {
        _ConnDone[i] = false;
    }

    // Incoming address
    struct sockaddr_in InAddr;
    socklen_t InAddrSize;
    ssize_t NByte;

    // Incoming packet
    uRequest Req;
    memset(&Req, 0, sizeof(Req));

    // Listen for data requests
    do {
        // Receive message
        InAddrSize = sizeof(struct sockaddr_in);
        NByte = recvfrom(_SockListen, &Req, sizeof(Req), 0,
                            (struct sockaddr *) &InAddr, &InAddrSize);

        if(NByte > 0) {
            if(Req.Type != UREQ_NEW && Req.Type != UREQ_CNT) {
                setError("Data request with wrong type: %hu. (in RBCServer::Send)",
                         Req.Type);
                return -1;
            }

            cout << "Received packet." << endl;

            // Connection manager
            int iConn = _nConnections - 1;
            for(; iConn > -1 && Req.senderID != _Connections[iConn]; --iConn){}

            // Append new connection
            if(iConn < 0) {
                if(_nConnections == MAX_CONN) {
                    for(unsigned int i = 1; i < MAX_CONN; i++) {
                        _outData[i-1] = _outData[i];
                        _iOutData[i-1] = _iOutData[i];
                        _ConnDone[i-1] = _ConnDone[i];
                    }
                } else {
                    _nConnections++;
                }
                iConn = _nConnections - 1;
                _outData[iConn] = 0;
                _iOutData[iConn] = 0;
                _ConnDone[iConn] = false;
                _Connections[iConn] = Req.senderID;
            }

            if(!_ConnDone[iConn]) {
                _ConnDone[iConn] = true;
                if(SendToConnection(iConn, &InAddr, Req) < 0) {
                    cout << "Failure." << endl;break;
                }
            }
        }

    } while(NByte >= 0);

    // Only EAGAIN or EWOULDBLOCK are acceptable values for errno
    if (errno != EWOULDBLOCK && errno != EAGAIN) {
        setError("Error %i on recvfrom: %s. (in RBCServer::Send)",
                 errno, strerror(errno));
        return -1;
    }

    return 0;
}

int RBCServer::SendToConnection(int iConn, sockaddr_in* OutAddr, uRequest& Req) {

    if(iConn >= MAX_CONN) {
        setError("Error: iConn >= MAX_CONN. (in RBCServer::SendToConnection)");
        return -1;
    }

    // Reasons for requesting new datafile
    bool reqNew = (Req.Type == UREQ_NEW && _outData[iConn] == 0);

    // Prepare data for "New"-request
    if(reqNew) {
        if(_outData[iConn]) delete [] _outData[iConn];

        // Prepare data
        _iOutData[iConn] = 0;
        if(Prepare(_outData[iConn], Req.resReq, Req.imgType) < 0) {
            setError("Error: Data prepare failed. (in RBCServer::Send)");
            return -1;
        }
    }

    // Send data
    if(_outData[iConn] != 0) {

        // Send all data that was prepared
        int NumPackets = _outData[iConn][0].H.NumPackets;
        for(; _iOutData[iConn] < NumPackets; ++_iOutData[iConn]) {

            // Send packet
            uPacket* outPacket = &(_outData[iConn][_iOutData[iConn]]);
            int outDataSize = sizeof(*outPacket);
            socklen_t AddrSize = sizeof(struct sockaddr_in);
            ssize_t NByte = sendto(_SockListen, outPacket, outDataSize,
                              0, (sockaddr*) OutAddr, AddrSize);

            if(NByte != outDataSize) {
                if (errno != EWOULDBLOCK && errno != EAGAIN) {
                    setError("Error %i on sendto: %s. %i/%i byte transmitted. (in RBCServer::Send)",
                             errno, strerror(errno), NByte, outDataSize);
                    return -1;
                }

                return 0;
            }
        }

        // Clear data if all transferred
        if(_iOutData[iConn] >= NumPackets) {
            delete [] _outData[iConn];
            _outData[iConn] = 0;
        }
    }

    return 0;
}

void RBCServer::Clear()
{
    for(int iConn = 0; iConn < MAX_CONN; ++iConn) {
        if(_outData[iConn]) delete [] _outData[iConn];
        _outData[iConn] = 0;
        _iOutData[iConn] = 0;
    }
}


/******************************************
* RBC Client                              *
******************************************/
RBCClient::RBCClient(unsigned int Port, const char* HostName) : RBCServerClient(Port, HostName)
{
    // Somewhat random number
    _senderID = (unsigned int) abs(((time(NULL)*181)*((getpid()-83)*359))%104729);

    // Create socket
    _SockListen = socket(AF_INET, SOCK_DGRAM, 0);
    if (_SockListen < 0) {
        setError("Error opening socket. (in RBCClient::RBCClient)");
        return;
    }

    // Set flag to non-blocking

   fcntl(_SockListen, F_SETFL, fcntl(_SockListen, F_GETFL) | O_NONBLOCK);

    // Increase buffer size
    long int n = UDPBUF_SIZE;
    setsockopt(_SockListen, SOL_SOCKET, SO_RCVBUF, &n, sizeof(n));

    // Find server address
    struct hostent *Host;
    Host = gethostbyname(_HostName);
    if(Host == 0) {
        //setError("Unknown host: %s. (in RBCClient::RBCClient)", _HostName);
        return;
    }

    _ServerAddr.sin_family = AF_INET;
    bcopy((char *)Host->h_addr,
         (char *)&_ServerAddr.sin_addr,
          Host->h_length);//was: memmove
    _ServerAddr.sin_port = htons(_Port);

    _Connected = true;
}

RBCClient::~RBCClient()
{
    close(_SockListen);
}
//ReqType reqType 0 newReq 1 cntReq 2 other

/*int RBCClient::Receive(net_dataSend* N, void *outData,ReqType reqType, uRequest::ImgType imgType, unsigned short resReq) { //ReqType
//int opts = fcntl(_SockListen,F_GETFL);
        // opts = opts & (~O_NONBLOCK);
    if(!_Connected) {
    cout << "Not connected." << endl;
   // break;


        return 0;
    }

    ssize_t NByte;
    unsigned int AddrSize;
    struct sockaddr_in ClientAddr;
if(reqType == newReq || reqType == cntReq ) {
    // Send data request
    if(reqType == newReq || reqType == cntReq) {
        uRequest Req;
        Req.Type = (reqType ==newReq)?(UREQ_NEW):(UREQ_CNT);
        Req.senderID = _senderID;
        Req.resReq = resReq;
        Req.imgType = imgType;

        AddrSize = sizeof(struct sockaddr_in);
        NByte = sendto(_SockListen, &Req, sizeof(Req), 0,
                       (const struct sockaddr *) &_ServerAddr, AddrSize);

        if (NByte < 0) {
            setError("Error %i on sendto: %s. (in RBCClient::Receive)",
                     errno, strerror(errno));
            return -1;
        }
    }*/
int RBCClient::Receive(NetUserInput* N, void *outData,ReqType reqType, uRequest::ImgType imgType, unsigned short resReq) {
int opts = fcntl(_SockListen,F_GETFL);
    opts = opts & (~O_NONBLOCK);
    if(!_Connected) {
        return 0;
    }

    ssize_t NByte;
    unsigned int AddrSize;
    struct sockaddr_in ClientAddr;

    // Send data request
    if(reqType == newReq || reqType == cntReq) {
        uRequest Req;
        Req.Type = (reqType == newReq)?(UREQ_NEW):(UREQ_CNT);
        Req.senderID = _senderID;
        Req.resReq = resReq;
        Req.imgType = imgType;
		Req.netUserInput = *N;
        AddrSize = sizeof(struct sockaddr_in);
        NByte = sendto(_SockListen, &Req, sizeof(Req), 0,
                       (const struct sockaddr *) &_ServerAddr, AddrSize);

        if (NByte < 0) {
            setError("Error %i on sendto: %s. (in RBCClient::Receive)",
                     errno, strerror(errno));
            return -1;
        }
    }

    // Receive data packets
    while(1) {

        uPacket Pack;
        AddrSize = sizeof(struct sockaddr_in);
        NByte = recvfrom(_SockListen, &Pack, sizeof(Pack), 0,
                         (struct sockaddr *) &ClientAddr, &AddrSize);

        if (NByte < 0) {

            //errno is set to EAGAIN or EWOULDBLOCK
            if((errno != EWOULDBLOCK) && (errno != EAGAIN)) {
                setError("Error %i on recvfrom: %s. (in RBCClient::Receive)",
                         errno, strerror(errno));
                return -1;
            }
            break;
        } else if(NByte != sizeof(Pack)) {
            setError("Error on recvfrom: %i/%i bytes read. (in RBCClient::Receive)",
                     NByte, sizeof(Pack));
            return -1;
        }

        Add(Pack, outData);


//cout<<*Pack.P.Data<<endl;}
//cout << Pack.P.Data[500] <<endl;
       break;
            }

    return 0;

}


