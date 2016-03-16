#ifndef PATCHTRACKER_H
#define PATCHTRACKER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "global_var.h"

#define PATCH_SIZE      9
#define PATCH_SIZE_MAX  17
#define NB_FEATURE_MAX  120


class Patch;

class PatchTracker
{

public:
    cudaArray* listPatchsDevice;      // Store the warped patch
    cudaArray* listPatchsMaxDevice;   // Store patch to be warped
    u_int8_t * ptlistPatchsMaxHost;
    cudaChannelFormatDesc channelDesc;
    int indiceFeatures;

    float2 *ptPositionFeaturesHost;
    float2 *ptPositionFeaturesDevice;

//public:
    PatchTracker();
   // addWarpedPatch(Patch *pt);
    void addPatchToWarp(u_int8_t * ptImage,int row,int col,float px,float py);
    void resetTracker(){indiceFeatures = 0;}
    void runWarp(void);



};

#endif // PATCHTRACKER_H
