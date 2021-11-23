//#include <string.h>
//#include <sys/types.h>
//#include <sys/socket.h>
//#include <netinet/tcp.h>
//#include <arpa/inet.h>
//#include <unistd.h>
//#include <errno.h>
//#include <string.h>
//#include <time.h> 
//#include <netdb.h>
//#include <math.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <unistd.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>


//my includes

//#include "output/matrix_folder_writer.h"
#include "output/datastruct.h"
#include "output/server_writer.h"
#include "output/serverclient.h"
#include "output/datafile.h"
//#include "output/datapoint_writer.h"
//#include "output/track_writer.h"



unsigned int port=1234;

uPacket upac;
    DataFolder ReadData;


//uPacket* pointer;
unsigned int a;
const char* HostName="localhost";
int main() {
uRequest::ImgType imgtype =uRequest::Temperature;
RBCClient::ReqType req=RBCClient::newReq;
DataFolderClient hehe(port,HostName);
unsigned short resReq=0;	
//int reqtype=0;
//hehe.RBCClient::Receive(upac,reqtype, imgType, resReq);
int cpt=0;
ofstream f_out ("Fichier.txt");
unsigned int ms=50000;

//usleep(ms);
while(1){
hehe.Receive(&ReadData,req, imgtype, 512);


//f_out<<*upac.P.Data<<endl;
//cout<<*upac.P.Data<<endl;

//cout<<*upac.P.Data<<upac.P.iStart<<upac.P.iEnd<<endl;
//for(int i(0); i<6; i++){



//cout<<upac.P.Data[i]<<";";
//f_out<<upac.P.Data[i]<<";";
//}
//if(ReadData.GetBest()->Nt()<65){
//if(ReadData.GetBest()!=0){
//cout<<ReadData.GetBest()->Nt()<<"/"<<ReadData.GetBest()->Nx()<<"/"<<ReadData.GetBest()->Ny()<<endl;
//int res=ReadData.GetBest()->Ny()*ReadData.GetBest()->Nx();



cout<<ReadData.GetBest()->Nt();
}}
//f_out<<cout<<ReadData.GetBest()->Nt()<<"        "<<ReadData.GetBest()->Data()[i]<<endl;
//}

//cout<<ReadData.GetBest()->Nt()<<endl;

//}
//}



