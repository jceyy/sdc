#ifndef COLORING_H_INCLUDED
#define COLORING_H_INCLUDED

#include <iostream>
#include <cstdint>

void grayScale1(float minCoeff, float maxCoeff, float value, Uint8* r, Uint8* g, Uint8* b);

void grayScale2(float minCoeff, float maxCoeff, float value, Uint8* r, Uint8* g, Uint8* b);

void colorScale1(float minCoeff, float maxCoeff, float value, Uint8 *r, Uint8 *g, Uint8 *b);

void colorScale2(float minCoeff, float maxCoeff, float value, Uint8 *r, Uint8 *g, Uint8 *b);

void colorScale3(float minCoeff, float maxCoeff, float value, Uint8* r, Uint8* g, Uint8* b);

void colorScale4(float minCoeff, float maxCoeff, float value, Uint8* r, Uint8* g, Uint8* b);

#endif // COLORING_H_INCLUDED
