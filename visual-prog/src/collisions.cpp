#include <iostream>
#include <SDL2/SDL.h>

bool rectCollision(SDL_Rect const& rect1, SDL_Rect const& rect2)
{
	if ( (box2.x >= box1.x + box1.w) || (box2.x + box2.w <= box1.x) || (box2.y >= box1.y + box1.h) || (box2.y + box2.h <= box1.y) )
		return false;
    	else
        	return true;
}

bool rectPointCollision(SDL_Rect const& box, int px, int py)
{
    if (px >= box.x && px < box.x + box.w && py >= box.y && py < box.y + box.h)
        return true;
    else
        return false;
}

