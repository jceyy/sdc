#include <iostream>
#include <SDL2/SDL.h>
using namespace std;

void grayScale1(float minCoeff, float maxCoeff, float value, Uint8* r, Uint8* g, Uint8* b)
{
    *r = *g = *b = (Uint8) 255.0f*(value-minCoeff)/(maxCoeff-minCoeff);
}

void grayScale2(float minCoeff, float maxCoeff, float value, Uint8* r, Uint8* g, Uint8* b)
{
    float temp = (value-minCoeff)/(maxCoeff-minCoeff);
    (*r) = (*g) = (*b) = (Uint8) 255.0f*temp*temp;
}

void colorScale1(float minCoeff, float maxCoeff, float value, Uint8 *r, Uint8 *g, Uint8 *b)
{
    float temp = (value-minCoeff)/(maxCoeff-minCoeff);
    (*r) = (Uint8) 255*temp;
    (*b) = (Uint8) 255-0.6* (*r);
    (*g) = (Uint8) (*r)*temp;
}

void colorScale2(float minCoeff, float maxCoeff, float value, Uint8 *r, Uint8 *g, Uint8 *b)
{
    float temp = (value-minCoeff)/(maxCoeff-minCoeff);
    (*r) = (Uint8) 255*temp*temp;
    temp = (value-maxCoeff)/(maxCoeff-minCoeff);
    (*b) = (Uint8) 255*temp*temp;
    (*g) = (Uint8) (*r+*b)/2;
}

void colorScale3(float minCoeff, float maxCoeff, float value, Uint8* r, Uint8* g, Uint8* b)
{
    float temp = (value-minCoeff)/(maxCoeff-minCoeff);
    (*r) = (*g) = (Uint8) 255*temp*temp;
    (*b) = 128;
}

void colorScale4(float minCoeff, float maxCoeff, float value, Uint8* r, Uint8* g, Uint8* b)
{
    float temp = (value-minCoeff)/(maxCoeff-minCoeff);
    (*r) = (Uint8) 255*temp*temp;
    (*g) = (Uint8)*(r)/10;
    (*b) = (Uint8)*(r)/10;
}
