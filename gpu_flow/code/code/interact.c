#include <interact.h>

#define SERV_TCP_PORT 8010
#define MAX_SIZE 8000

void Viztosim(net_dataSend N, int SERV_TCP_PORT, int MAX_SIZE)
{
  int sockfd, newsockfd, servlen, ssockfd, newssockfd;
  struct sockaddr_in serv_addr;
  char *serv_host = "localhost";
  struct hostent *host_ptr;
  int port;
  int buff_size = 0;
  char string[MAX_SIZE];
  int len;
struct sockaddr_in cli_addr;

  /* command line: client [host [port]]*/
  if(argc >= 2)
     serv_host = argv[1]; /* read the host if provided */
  if(argc == 3)
     sscanf(argv[2], "%d", &port); /* read the port if provided */
  else
     port = SERV_TCP_PORT;

  /* get the address of the host */
  if((host_ptr = gethostbyname(serv_host)) == NULL) {
     perror("gethostbyname error");
     exit(1);
  }

  if(host_ptr->h_addrtype !=  AF_INET) {
     perror("unknown address type");
     exit(1);
  }

  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr =
     ((struct in_addr *)host_ptr->h_addr_list[0])->s_addr;
  serv_addr.sin_port = htons(port);


  /* open a TCP socket */
  if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
     perror("can't open stream socket");
     exit(1);
  }

  /* connect to the server */
  if(connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
     perror("can't connect to server");
     exit(1);
  }

 /* open a TCP socket (an Internet stream socket) */
  if((ssockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
     perror("can't open stream socket");
     exit(1);
  }
  if(bind(ssockfd, (struct sockaddr *) &cli_addr, sizeof(cli_addr)) < 0) {
     perror("can't bind local address");
     exit(1);
  }
std::string rad= "r="+ std::to_string(N.radius)+"/pX=" + std::to_string(N.posX)+"/pY=" + std::to_string(N.posY)+"/m=" + std::to_string(N.mode)+"/T=" + std::to_string(N.temp)+"/velX=" + std::to_string(N.velX)+"/velY=" + std::to_string(N.velY)+"/resX=" + std::to_string(N.resX)+"/resY=" + std::to_string(N.resY)+"/"+"@";
//data to string chain
char szBuffer[strlen(dat)];
strcpy(szBuffer, dat.c_str());

write(sockfd, szBuffer , sizeof("szBuffer"));
close(sockfd);

}

void Simread(net_dataSend &N, int MAX_SIZE, int SERV_TCP_PORT, int WAIT_TIME)
{
  int sockfd, newsockfd, clilen;
  struct sockaddr_in cli_addr, serv_addr;
  int port;
  char string[MAX_SIZE];
  int len;
  if(argc == 2)
     sscanf(argv[1], "%d", &port); /* read the port number if provided */
  else
     port = SERV_TCP_PORT;
/* open a TCP socket (an Internet stream socket) */
  if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
     perror("can't open stream socket");
     exit(1);
  }
  /* bind the local address, so that the cliend can send to server */
  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  serv_addr.sin_port = htons(port);
  if(bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
     perror("can't bind local address");
     exit(1);
  }
  /* open a TCP socket */
  if((newssockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
     perror("can't open stream socket");
     exit(1);
  }
  if(connect(newssockfd, (struct sockaddr *) &cli_addr, sizeof(cli_addr)) < 0) {
     perror("can't connect to client");
     exit(1);
  }

 int cpt;
 while(cpt<WAIT_TIME) {
    SDL_Delay(1);
    cpt=cpt+1
  /* listen to the socket */
  listen(sockfd, 5);
 /* wait for a connection from a client; this is an iterative server */
     clilen = sizeof(cli_addr);
     newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);

     if(newsockfd < 0) {
        perror("can't bind local address");
     }

     len = read(newsockfd, buffer, MAX_SIZE);
int cpt1=0;
     string[len] = 0;
     for (int i (0); i<sizeof(buffer),i++ ){
           int j=1;
        if (buffer[i]=="="){
            while(buffer[i+j]!="/"){
                    j=j+1;
                    std::string str;
            str.push_back(buffer[i+j]);}
    cpt1=cpt1+1;

    switch(cpt1){
    case 0: N.radius=stoi(str);
    case 1: N.posX=stoi(str);
    case 2: N.posY=stoi(str);
    case 3: N.mode=stoi(str);
    case 4: N.temp=stoi(str);
    case 5: N.velX=stoi(str);
    case 6: N.velY=stoi(str);
    case 7: N.resX=stoi(str);
    case 8: N.resY=stoi(str);}
     }
 if (buffer[i]=="@") break;

   }
close(sockfd);
}

  }


