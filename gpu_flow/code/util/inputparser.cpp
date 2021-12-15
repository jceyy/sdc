#include "inputparser.h"

#include <string>
#include <string.h>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <typeinfo>


/***********************************************************************************/
/** Input parser class                                                            **/
/***********************************************************************************/
inputparser::inputparser(int argc, char **argv) : _argc(argc), _argv(argv) {}

// Standalone functions
// opt
bool inputparser::getopt(const char *name)
{
    cout << "Option: " << name << "(optional)" << endl;
    return (pos(name) >= 0);
}

// N chars
void inputparser::getstring(char* out, int string_size, const char *name, isMandatory m)
{
    char *stringlist[1] = {out};
    getstrings(stringlist, string_size, 1, name, m);
}

// N strings
void inputparser::getstrings(char** out, int string_size, int count, const char *name, isMandatory m)
{
    bool found = false;
	int argmax = _argc - count;

    for(int argi = 0; argi < argmax; ++argi) {
        if(strcmp(_argv[argi], name) == 0) {
            found = true;
            for(int n = 0; n < count; n++) {
                strncpy(out[n], _argv[argi+n+1], string_size - 1);
                out[n][string_size - 1] = 0;
            }
        }
    }

    cout << "Option: " << name << " <" << count << " strings> ";
    if(m == mandatory) cout << "(mandatory)" << endl;
    else cout << "(optional)" << endl;

    if(m == mandatory && !found) {
        EXIT_ERROR2("Mandatory argument not found: ", name);
    }

}

// Get one argument from argument list
template<class T>
void inputparser::get(T& out, const char *name, inputparser::isMandatory m) {

	vector<T> outvec;
	get(outvec, 1, name, m);

	if(outvec.size() > 0) {
		out = outvec.at(0);
	}
}

// Get list of arguments from argument list. count == -1 extracts until end of argv.
// Number of extractions has to be specified.
template<class T>
void inputparser::get(std::vector<T>& out, int count, const char *name, inputparser::isMandatory m) {

    // Find keyword
    int argi = pos(name);
    bool found = (argi >= 0);

    // Increment argi to first real argument (after key)
    ++argi;

    // Special count = -1 case
    if(count < 0) {
        count = _argc - argi;
    }

    // Error if less found than mandatorily needed
    if(count > _argc - argi) {
        found = false;
    }

    // Find parameters
    if(found) {

        out.resize(count);

        for(int i = 0; i < count; ++i) {
            std::stringstream s;
	    T val;
            s.str(_argv[argi + i]);
            s >> val;

            if(s.fail()){
                EXIT_ERROR2("Parsing argument failed: ", name);
		}

	    out.at(i) = val;
        }
    }

    cout << "Option: " << name << " <" << out.size() << "*" << typeid(T).name() << "> ";
    if(m == mandatory) cout << "(mandatory)" << endl;
    else cout << "(optional)" << endl;

    if(m == mandatory && !found) {
        EXIT_ERROR2("Mandatory argument not found: ", name);
    }
}

void inputparser::print() {
    for(int i = 0; i < _argc; ++i) {
        cout << _argv[i] << ' ';
    }
    cout << endl;
}

// Get first position of key in argv. Return -1 if not found.
int inputparser::pos(const char *name) {

    for(int argi = 0; argi < _argc ; ++argi) {
        if(strcmp(_argv[argi], name) == 0) {
            return argi;
        }
    }

	return -1;
}

// Implementations
template void inputparser::get<int>(int& out, const char *name, inputparser::isMandatory m);
template void inputparser::get<int>(std::vector<int>& out, int count, const char *name, inputparser::isMandatory m);
template void inputparser::get<float>(float& out, const char *name, inputparser::isMandatory m);
template void inputparser::get<float>(std::vector<float>& out, int count, const char *name, inputparser::isMandatory m);
template void inputparser::get<double>(double& out, const char *name, inputparser::isMandatory m);
template void inputparser::get<double>(std::vector<double>& out, int count, const char *name, inputparser::isMandatory m);



