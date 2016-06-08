#ifndef PATCHTRACKER_H
#define PATCHTRACKER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "global_var.h"




class Patch;

class PatchTracker
{

public:

    /// Device variables
    u_int8_t*  u8_PatchsWithBorder_Device;  // Store the warped patch on device
    cudaArray* Array_PatchsMax_Device;         // Store patchs to be warped on device
    cudaChannelFormatDesc channelDesc;
    float2 *f2_PositionFeatures_Device;       // Store the position (u,v) of the feature to track on device
    float *f_Matrix_Device;

    /// Host variables
    u_int8_t *u8_ListPatchsMax_Host;          // Store patchs max on host
    u_int8_t *u8_ListPatchsWithBorder_Host;   // Store patchs with border on host
    float2 *f2_PositionFeatures_Host;         // Store the position (u,v) of the feature to track on host
    int i_IndiceFeaturesToWarp;              // Store the number of Patch which add to be warped
    float *f_Matrix_Host;

    float2 *ftmp_Device;
    float2 *ftmp_Host;

//public:
    PatchTracker();
    void addPatchToWarp(u_int8_t * ptImage,int row,int col,float px,float py);
    void resetTracker(){i_IndiceFeaturesToWarp = 0;}
    void runWarp(void);

};

#endif // PATCHTRACKER_H
