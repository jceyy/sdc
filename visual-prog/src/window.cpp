#include <SDL2/SDL.h>
#include <SDL_ttf.h>
#include <SDL_image.h>
#include <iostream>

int prepareWindow(SDL_Window** window, SDL_Renderer **renderer, std::string const& title)
{
	SDL_Init(SDL_INIT_EVERYTHING);
	*window = SDL_CreateWindow(title.c_str(), 0, 0, 1280, 720, SDL_WINDOW_RESIZABLE);
	if (window == nullptr)
		return -1;
		
	*renderer = SDL_CreateRenderer(*window, -1, 0);
	if (*renderer == nullptr)
		return -1;
	
	return 0;
}

void freeWindow(SDL_Window* window, SDL_Renderer *renderer)
{
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	IMG_Quit();
	TTF_Quit();
	SDL_Quit();
}


