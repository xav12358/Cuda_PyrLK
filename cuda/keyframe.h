#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <vector>


#include "global_var.h"





class Level
{

    u_int8_t *ptData;
    int irows,icols;
    short* vCorners;
    short* vCornersMax;
    int*   mScore;

public:
    Level(){};
    Level(int rows,int cols);
//    Level& operator=(const Level &rhs);
};


class KeyFrame
{
    Level Levels[4];
public:

    KeyFrame();
    ~KeyFrame();

};

#endif // KEYFRAME_H
