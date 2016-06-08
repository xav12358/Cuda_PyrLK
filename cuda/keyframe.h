#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <vector>


#include "global_var.h"

typedef enum{
    ToCalculate,
    ToStore
}KeyFrameType;

class Level
{

public:

    int irow,icol;

    u_int8_t *u8_ptData_Host; ///< Store the image data

    // Max KeyPoints location
    short2* kpLoc_Device;
    short2* kpLocMax_Device;
    float* scoreMax_Host;

    int iNbKeypoints;   ///< Number of keypoints
    int iNbKeypointsMax;   ///< Number of keypoints
    int threshold;      ///< Theshold at this level

    Level(){};
    void BuildLevel(int rows,int cols);
};


class KeyFrame
{
    int irow,icol;
    KeyFrameType eKFType;   ///< Store the type of the keyframe (not allocate memory on gpu for storetype keyframe)

    // Cuda memory location for the image
    u_int8_t *u8_ptDataIn_Device;      ///< Image In
    u_int8_t *u8_ptDataOut_Device;     ///< Image Out
    u_int8_t *u8_ptTmpBloc_Device;     ///< Image Tmp


    // Cuda memory location for the keypoints and maxkeypoint location
    short2* kpLoc_Device;
    int* vCorners_Device;

    short2* kpLocMax_Device;
    float* vCornersMax_Device;


public:



    Level Levels[4];

    KeyFrame(int row,int col,KeyFrameType eKFType);
    ~KeyFrame();

    void MakePyramid(u_int8_t *u8_ptDataSrc);



};

#endif // KEYFRAME_H
