#include "patchtracker.h"


texture<unsigned char, 2,cudaReadModeElementType> PatchListInMaxWithBorder;
texture<unsigned char, 2,cudaReadModeElementType> PatchListOut;


__global__ void WarpPatch (u_int8_t *ptGrayIn, int w, int h)
{

}


PatchTracker::PatchTracker():
    indiceFeatures(0)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    channelDesc = cudaCreateChannelDesc<u_int8_t>();

    // Create data in GPU memory
    checkCudaErrors(cudaMallocArray(&listPatchsDevice, &channelDesc, PATCH_SIZE, PATCH_SIZE*NB_FEATURE_MAX));
    checkCudaErrors(cudaMallocArray(&listPatchsMaxDevice, &channelDesc, PATCH_SIZE_MAX, PATCH_SIZE_MAX*NB_FEATURE_MAX));
    checkCudaErrors(cudaBindTexture2D(0,&PatchListOut,listPatchsDevice,&desc,PATCH_SIZE_MAX , PATCH_SIZE_MAX*NB_FEATURE_MAX ,PATCH_SIZE_MAX));

    // Create data in Host memory
    ptlistPatchsMaxHost = (u_int8_t*)malloc(PATCH_SIZE_MAX*PATCH_SIZE_MAX*NB_FEATURE_MAX*sizeof(u_int8_t));

    // Create feature location
    ptPositionFeaturesHost = (float2*)malloc(NB_FEATURE_MAX*sizeof(float2));
    checkCudaErrors(cudaMalloc((void **)&ptPositionFeaturesDevice,  NB_FEATURE_MAX * sizeof(float2)));



}


//////////////////////////
/// \brief PatchTracker::addWarpedPatch
/// \param pt
///
void PatchTracker::addPatchToWarp(u_int8_t * ptImage,int row,int col,float px,float py)
{
    u_int8_t * ptPatchMaxHost;
    int Step =  PATCH_SIZE_MAX*PATCH_SIZE_MAX;

    ptPositionFeaturesHost[indiceFeatures].x = px;
    ptPositionFeaturesHost[indiceFeatures].y = py;

    int indexCeilx = ceil(px)-8;
    int indexCeily = ceil(py)-8;
    for(int y = 0;y<PATCH_SIZE_MAX;y++)
    {
        ptPatchMaxHost =  ptlistPatchsMaxHost + PATCH_SIZE_MAX*y  + Step*indiceFeatures;
        for(int x = 0;x<PATCH_SIZE_MAX;x++,ptPatchMaxHost++)
        {
            *ptPatchMaxHost  =  ptImage[indexCeilx+x + (indexCeily+y)*col];
        }
    }
    indiceFeatures++;
}


///////////////////////////
/// \brief runWarp
///
void PatchTracker::runWarp(void)
{

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();

    checkCudaErrors(cudaMemcpy2DToArray( listPatchsMaxDevice,
                     0,
                     0,
                     ptlistPatchsMaxHost,
                     PATCH_SIZE_MAX,
                     PATCH_SIZE_MAX*NB_FEATURE_MAX,
                     PATCH_SIZE_MAX,
                     cudaMemcpyHostToDevice  ));

    // Bind the texture to the array
    checkCudaErrors(cudaBindTexture2D(0,&PatchListInMaxWithBorder,listPatchsMaxDevice,&desc,PATCH_SIZE_MAX , PATCH_SIZE_MAX*NB_FEATURE_MAX ,PATCH_SIZE_MAX));


}


