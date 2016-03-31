#include "patchtracker.h"


texture<unsigned char, 2,cudaReadModeElementType> PatchListInMax;
//texture<float, 2,cudaReadModeElementType> PatchListOut;


__global__ void WarpPatch (u_int8_t * ptOutPatch, float *Matrix,float2* f_pos,float2 *ftmp)
{

    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;

    //    __shared__ float2 f_PosPixel;
    __shared__ float2 f_deltaPixel;
    __shared__ int iPatchIndex;

    if(threadIdx_x == 0 && threadIdx_y == 0)
    {
//        ftmp[threadIdx_x+threadIdx_y*11].x = blockIdx.x;

        //ftmp[blockIdx.x].x = PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER* blockIdx.x;
        //        f_deltaPixel = f_pos[blockIdx.x] - ceil(f_pos[blockIdx.x] ) ;
        iPatchIndex = PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER* blockIdx.x;    // index to the current patch
    }
    syncthreads();


    float2 index_PatchMax;
    index_PatchMax.x = (threadIdx_x-HALF_PATCH_SIZE_WITH_BORDER) *Matrix[0]  + (threadIdx_y-HALF_PATCH_SIZE_WITH_BORDER) *Matrix[1] + PATCH_MAX_CENTER;//+f_deltaPixel.x;
    index_PatchMax.y = (threadIdx_x-HALF_PATCH_SIZE_WITH_BORDER) *Matrix[2]  + (threadIdx_y-HALF_PATCH_SIZE_WITH_BORDER) *Matrix[3] + PATCH_MAX_CENTER;//+f_deltaPixel.y;
    index_PatchMax.y += (blockIdx.x)*PATCH_SIZE_MAX;


    if(blockIdx.x == 0)
    {
//        ftmp[threadIdx_x+threadIdx_y*11].x = index_PatchMax.x;
//        ftmp[threadIdx_x+threadIdx_y*11].y = index_PatchMax.y;
    }
    float J_val = tex2D(PatchListInMax,index_PatchMax.x,index_PatchMax.y);

    ptOutPatch[threadIdx_x+threadIdx_y*PATCH_SIZE_WITH_BORDER + iPatchIndex] = (u_int8_t)J_val;
}


PatchTracker::PatchTracker():
    i_IndiceFeaturesToWarp(0)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    channelDesc = cudaCreateChannelDesc<u_int8_t>();

    // Create data in GPU memory
//    checkCudaErrors(cudaMallocArray(&u8_PatchsWithBorderDevice, &channelDesc, PATCH_SIZE_WITH_BORDER, PATCH_SIZE_WITH_BORDER*NB_FEATURE_MAX));
    checkCudaErrors(cudaMalloc((void **)&u8_PatchsWithBorderDevice,  PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER*NB_FEATURE_MAX * sizeof(u_int8_t)));


    checkCudaErrors(cudaMallocArray(&Array_PatchsMaxDevice, &channelDesc, PATCH_SIZE_MAX, PATCH_SIZE_MAX*NB_FEATURE_MAX));
    //    checkCudaErrors(cudaBindTexture2D(0,&PatchListOut,listPatchsDevice,&desc,PATCH_SIZE_MAX , PATCH_SIZE_MAX*NB_FEATURE_MAX ,PATCH_SIZE_MAX));

    // Create data in Host memory
    u8_ListPatchsMaxHost        = (u_int8_t*)malloc(PATCH_SIZE_MAX*PATCH_SIZE_MAX*NB_FEATURE_MAX*sizeof(u_int8_t));
    u8_ListPatchsWithBorderHost = (u_int8_t*)malloc(PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER*NB_FEATURE_MAX*sizeof(u_int8_t));

    // Create feature location
    f2_PositionFeaturesHost = (float2*)malloc(NB_FEATURE_MAX*sizeof(float2));
    checkCudaErrors(cudaMalloc((void **)&f2_PositionFeaturesDevice,  NB_FEATURE_MAX * sizeof(float2)));


    // Create matrix
    f_MatrixHost = (float*)malloc(4*sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&f_MatrixDevice,  4 * sizeof(float)));

    //tmp matrix
    ftmpHost        = (float2*)malloc(PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER*sizeof(float2));
    checkCudaErrors(cudaMalloc((void **)&ftmpDevice,  PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER * sizeof(float2)));

}


//////////////////////////
/// \brief PatchTracker::addWarpedPatch
/// \param pt
///
void PatchTracker::addPatchToWarp(u_int8_t * ptImage,int row,int col,float px,float py)
{
    u_int8_t * ptPatchMaxHost;
    int Step =  PATCH_SIZE_MAX*PATCH_SIZE_MAX;

    f2_PositionFeaturesHost[i_IndiceFeaturesToWarp].x = px;
    f2_PositionFeaturesHost[i_IndiceFeaturesToWarp].y = py;

    int indexCeilx = ceil(px)-8;
    int indexCeily = ceil(py)-8;
    for(int y = 0;y<PATCH_SIZE_MAX;y++)
    {
        ptPatchMaxHost =  u8_ListPatchsMaxHost + PATCH_SIZE_MAX*y  + Step*i_IndiceFeaturesToWarp;
        for(int x = 0;x<PATCH_SIZE_MAX;x++,ptPatchMaxHost++)
        {
            //if(x>3 && x<13 && y>3 && y<13)
                *ptPatchMaxHost  =  ptImage[indexCeilx+x + (indexCeily+y)*col];
             //   *ptPatchMaxHost = 255;
            //else
             //   *ptPatchMaxHost = 0;

        }
    }
    i_IndiceFeaturesToWarp++;
}


///////////////////////////
/// \brief runWarp
///
void PatchTracker::runWarp(void)
{

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();

    checkCudaErrors(cudaMemcpy2DToArray( Array_PatchsMaxDevice,
                                         0,
                                         0,
                                         u8_ListPatchsMaxHost,
                                         PATCH_SIZE_MAX,
                                         PATCH_SIZE_MAX,
                                         PATCH_SIZE_MAX*i_IndiceFeaturesToWarp,
                                         cudaMemcpyHostToDevice  ));

    ////////////////////////////////
    // Bind the texture to the array
    //    checkCudaErrors(cudaBindTexture2D(0,&PatchListInMax,Array_PatchsMaxDevice,&desc,PATCH_SIZE_MAX , PATCH_SIZE_MAX*NB_FEATURE_MAX ,PATCH_SIZE_MAX));
    checkCudaErrors(cudaBindTextureToArray(PatchListInMax,Array_PatchsMaxDevice));


//    ////////////////////////////////
//    // copy the position of feature in image
//    checkCudaErrors( cudaMemcpy(f2_PositionFeaturesDevice, f2_PositionFeaturesHost, i_IndiceFeaturesToWarp*sizeof(float2), cudaMemcpyHostToDevice) );

    ////////////////////////////////
    // copy the position of feature in image
    checkCudaErrors( cudaMemcpy(f2_PositionFeaturesDevice, f2_PositionFeaturesHost, i_IndiceFeaturesToWarp*sizeof(float2), cudaMemcpyHostToDevice) );


    ////////////////////////////////
    // fill the Matrix
    f_MatrixHost[0] = 1.0;
    f_MatrixHost[1] = 0.0;
    f_MatrixHost[2] = 0.0;
    f_MatrixHost[3] = 1.0;
    checkCudaErrors( cudaMemcpy(f_MatrixDevice, f_MatrixHost, 4*sizeof(float), cudaMemcpyHostToDevice) );


    ////////////////////////////////
    /// Start the process
    dim3 blocks( i_IndiceFeaturesToWarp , 1);
    dim3 threads(PATCH_SIZE_WITH_BORDER, PATCH_SIZE_WITH_BORDER);
    WarpPatch<<<blocks,threads>>>(u8_PatchsWithBorderDevice,f_MatrixDevice,f2_PositionFeaturesDevice,ftmpDevice);


//    checkCudaErrors( cudaMemcpy(f2_PositionFeaturesHost, f2_PositionFeaturesDevice,PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER*sizeof(u_int8_t) , cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy(u8_ListPatchsWithBorderHost, u8_PatchsWithBorderDevice,PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER*i_IndiceFeaturesToWarp*sizeof(u_int8_t) , cudaMemcpyDeviceToHost) );
    //checkCudaErrors( cudaMemcpy(ftmpHost, ftmpDevice,PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER*sizeof(float2) , cudaMemcpyDeviceToHost) );


}


