#pragma once

#include <vector>
#include <stdio.h>
#include <stdlib.h>

#ifndef EXIT_ERROR
#define EXIT_ERROR(x) { fprintf(stderr, "Error at %s:%d: %s\n", __FILE__, __LINE__, x); exit(EXIT_FAILURE);}
#endif
#ifndef EXIT_ERROR2
#define EXIT_ERROR2(x,y) { fprintf(stderr, "Error at %s:%d: %s %s\n", __FILE__, __LINE__, x, y); exit(EXIT_FAILURE);}
#endif

using namespace std;

/***********************************************************************************/
/** Input parser class                                                            **/
/***********************************************************************************/
class inputparser
{
public:
    enum isMandatory { mandatory, optional };

    inputparser(int argc, char **argv);

    // get boolean option
    bool getopt(const char *name);

    // get one or several strings (char*s)
    void getstring(char *out, int string_size, const char *name, isMandatory m);
    void getstrings(char **out, int string_size, int count, const char *name, isMandatory m);

    // get one or two parameters (specially defined in .cu)
    template<class T>
    void get(T& out, const char *name, isMandatory m);
    template<class T>
    void get(std::vector<T>& out, int count, const char *name, isMandatory m);

    void print();

private:
	int pos(const char* name);

    int _argc;
    char **_argv;
};
