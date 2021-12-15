#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <netdb.h>
#include <stdarg.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <fcntl.h>


#define MAX_SIZE 800000

int main(int argc, char *argv[])
{
  int sockfd, newsockfd, servlen, ssockfd, newssockfd;
  struct sockaddr_in serv_addr;
struct sockaddr_in serv_addr2;
  char *serv_host = "localhost";
  struct hostent *host_ptr;
  int port;
  int buff_size = 0;
  char string[MAX_SIZE];
  int len;
struct sockaddr_in cli_addr;
struct sockaddr_in cli_addr2;




while(1){
#define SERV_TCP_PORT2 8033 
  /* command line: client [host [port]]*/
  if(argc >= 2) 
     serv_host = argv[1]; /* read the host if provided */
  if(argc == 3)
     sscanf(argv[2], "%d", &port); /* read the port if provided */
  else 
     port = SERV_TCP_PORT2;

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
  while(connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
     perror("can't connect to server");
exit(1);
     
  }
printf("Type your message here ");
char szBuffer[1256];
 gets(szBuffer);
 printf("\n\nszBuffer = %s\n",szBuffer);
write(sockfd, szBuffer , sizeof("szBuffer"));
close(sockfd);
}


}

