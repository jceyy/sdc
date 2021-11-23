#ifndef WINDOW_H_INCLUDED
#define WINDOW_H_INCLUDED

int prepareWindow(SDL_Window **window, SDL_Renderer **renderer, std::string const& title);
void freeWindow(SDL_Window* window, SDL_Renderer *renderer);

#endif // WINDOW_H_INCLUDED
