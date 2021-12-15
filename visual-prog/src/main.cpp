#include <SDL2/SDL.h>
#include <iostream>

#include "window.h"
#include "visual.h"
using namespace std;

int main(int argc, char *argv[])
{
	for (int i(0); i < argc; ++i) cout << argv[i] << " ";
	cout << endl;
	// Window creation
	SDL_Renderer *renderer = nullptr;
    	SDL_Window *window = nullptr;
    
    	int ret = prepareWindow(&window, &renderer, "visual-chaos");
	if (ret != 0) /* Initialize all the things we need */
	{
		cerr << "Failed to properly initialize" << endl;
		exit(-1);
	}

	ret = run(window, renderer);
	freeWindow(window, renderer);

	return ret;
}
