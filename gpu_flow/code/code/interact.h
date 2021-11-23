#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

#include <time.h>
#include <netdb.h>
#include <stdarg.h>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <sys/resource.h>
#include <fcntl.h>


/****************************************************************************
** This class manages the connection of the RBC GPU code to the UDP server
** in serverclient.h (which is part of RBCvisual).
****************************************************************************/
struct net_dataSend
{
    float radius; // modification radius
    int posX, posY; // where the mouse is
    int mode; // 1 = temperature, 2 = velocity,  0 (or anything else) = no modification
    float temp, velX, velY; // temperature to set
    int resX, resY; // actual resolution
};


class netw 
{
public:
    void Viztosim(net_dataSend N, int SERV_TCP_PORT, int MAX_SIZE);
    void Simread(net_dataSend &N, int MAX_SIZE, int SERV_TCP_PORT, int WAIT_TIME);
};