#ifndef PYRLK_H
#define PYRLK_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include "global_var.h"



class PyrDown_gpu;

class PyrLK_gpu
{

    PyrDown_gpu *ptPyrDownI;
    PyrDown_gpu *ptPyrDownJ;

    int iNbMaxFeaturesLK;

//    float *f2_PointsPrevHost;
    float2 *f2_PointsPrevDevice;

    float2 *f2_PointsNextHost;
    float2 *f2_PointsNextDevice;

    u_int8_t * u8_StatusHost;
    u_int8_t * u8_StatusDevice;

public:


    PyrLK_gpu(int rows, int cols, int iNbMaxFeatures = MAX_FEATURES_TO_SEARCH);
    ~PyrLK_gpu();
    void run_sparse(u_int8_t  *Idata, u_int8_t*Jdata, int h, int w, float2 *f2_Points, int iNbPoints);

};

#endif // PYRLK_H
