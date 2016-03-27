#ifndef PATCHTRACKER_H
#define PATCHTRACKER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "global_var.h"

#define PATCH_SIZE_WITH_BORDER                  11
#define HALF_PATCH_SIZE_WITH_BORDER             5
#define PATCH_SIZE_MAX                          17
#define PATCH_MAX_CENTER                        8
#define NB_FEATURE_MAX                          120


class Patch;

class PatchTracker
{

public:

    /// Device variables
    u_int8_t*  u8_PatchsWithBorderDevice;  // Store the warped patch on device
    cudaArray* Array_PatchsMaxDevice;         // Store patchs to be warped on device
    cudaChannelFormatDesc channelDesc;
    float2 *f2_PositionFeaturesDevice;       // Store the position (u,v) of the feature to track on device
    float *f_MatrixDevice;

    /// Host variables
    u_int8_t *u8_ListPatchsMaxHost;          // Store patchs max on host
    u_int8_t *u8_ListPatchsWithBorderHost;   // Store patchs with border on host
    float2 *f2_PositionFeaturesHost;         // Store the position (u,v) of the feature to track on host
    int i_IndiceFeaturesToWarp;              // Store the number of Patch which add to be warped
    float *f_MatrixHost;

    float2 * ftmpDevice;
    float2 * ftmpHost;

//public:
    PatchTracker();
    void addPatchToWarp(u_int8_t * ptImage,int row,int col,float px,float py);
    void resetTracker(){i_IndiceFeaturesToWarp = 0;}
    void runWarp(void);

};

#endif // PATCHTRACKER_H
