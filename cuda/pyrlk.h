#ifndef PYRLK_H
#define PYRLK_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include "global_var.h"



class PyrDown_gpu;
class PatchTracker;
class PyrLK_gpu
{

public:
    PyrDown_gpu *ptPyrDownI;
    PyrDown_gpu *ptPyrDownJ;

    int iNbMaxFeaturesLK;

//    float *f2_PointsPrevHost;
    float2 *f2_PointsPrev_Device;

    float2 *f2_PointsNext_Host;
    float2 *f2_PointsNext_Device;

    u_int8_t * u8_Status_Host;
    u_int8_t * u8_Status_Device;




    PyrLK_gpu(int rows, int cols, int iNbMaxFeatures = NB_FEATURE_MAX);
    ~PyrLK_gpu();
    void run_sparse(u_int8_t  *u8_ImagePrevHost, u_int8_t*u8_ImageNextHost, int h, int w, float2 *f2_PointsPrevHost, int iNbPoints);
    void run_sparsePatch(u_int8_t  *u8_ImagePrevDevice, PatchTracker *ptTracker, int h, int w);

};

#endif // PYRLK_H
