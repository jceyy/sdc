#ifndef COLLISION_H_INCLUDED
#define COLLISION_H_INCLUDED

bool rectCollision(SDL_Rect const& rect1, SDL_Rect const& rect2);
bool rectPointCollision(SDL_Rect const& box, int px, int py);

#endif // COLLISION_H_INCLUDED
