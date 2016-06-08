#include "fast.h"
#include "keyframe.h"
#include "pyrdown.h"

#define __RATIO__ 0.05

void Level::BuildLevel(int rows,int cols)
{

    icol = cols;
    irow = rows;

    checkCudaErrors(cudaMalloc((void **)&kpLoc_Device,  icol*irow*__RATIO__ * sizeof(short2)));
    //checkCudaErrors(cudaMalloc((void **)&vCorners_Device,  icol*irow*__RATIO__ * sizeof(int)));
    //checkCudaErrors(cudaMalloc((void **)&ptData,  rows*cols * sizeof(u_int8_t)));

    //checkCudaErrors(cudaMalloc((void **)&kpLocMax_Device,  icol*irow*__RATIO__ * sizeof(short2)));
    //checkCudaErrors(cudaMalloc((void **)&vCornersMax_Device,  icol*irow*__RATIO__ * sizeof(float)));


    u8_ptData_Host = (u_int8_t*)malloc(irow*icol*sizeof(u_int8_t));

    //kpLoc_Device = (short2*)malloc(irow*icol*__RATIO__*sizeof(short2));
    kpLocMax_Device = (short2*)malloc(irow*icol*__RATIO__*sizeof(short2));
    scoreMax_Host = (float*)malloc(irow*icol*__RATIO__*sizeof(float));

}





//////////////////////////////////////
/// \brief KeyFrame::KeyFrame
///
KeyFrame::KeyFrame(int row, int col, KeyFrameType eKFTypeL)
{
    icol = col;
    irow = row;

    eKFType = eKFTypeL;

    switch(eKFType)
    {
    case ToCalculate:
        // Create the tmpBloc for image downsampling
        checkCudaErrors(cudaMalloc((void **)&u8_ptDataIn_Device,  row*col * sizeof(u_int8_t)));
        checkCudaErrors(cudaMalloc((void **)&u8_ptDataOut_Device,  row*col * sizeof(u_int8_t)));
        checkCudaErrors(cudaMalloc((void **)&u8_ptTmpBloc_Device,  row*col * sizeof(u_int8_t)));

        checkCudaErrors(cudaMalloc((void **)&kpLoc_Device,  icol*irow*__RATIO__ * sizeof(short2)));
        checkCudaErrors(cudaMalloc((void **)&vCorners_Device,  icol*irow*__RATIO__ * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&kpLocMax_Device,  icol*irow*__RATIO__ * sizeof(short2)));
        checkCudaErrors(cudaMalloc((void **)&vCornersMax_Device,  icol*irow*__RATIO__ * sizeof(float)));
        break;

    case ToStore:
        break;
    }

    for(int i=0;i<4;i++)
    {
        Levels[i].BuildLevel(row>>i,col>>i);
    }
}




//////////////////////////////////
/// \brief KeyFrame::~KeyFrame
///
KeyFrame::~KeyFrame()
{

}



//////////////////////////////////
/// \brief KeyFrame::MakePyramid
/// \param pt
///


void KeyFrame::MakePyramid(u_int8_t *u8_ptDataSrc_Host)
{
    ///////////////////////////////////
    // Fill each level with the data

    int i_tabThreshold[4];
    i_tabThreshold[0]= 50;
    i_tabThreshold[1]= 40;
    i_tabThreshold[2]= 30;
    i_tabThreshold[3]= 20;
    //copy the first image
    checkCudaErrors(cudaMemcpy(u8_ptDataIn_Device  , u8_ptDataSrc_Host, irow * icol * sizeof(u_int8_t), cudaMemcpyHostToDevice));

    //Copy the source image to the first level
    int size = irow * icol/4;
    u_int32_t *u32_ptDataSrc = (u_int32_t *)(u8_ptDataSrc_Host);
    u_int32_t *u32_ptDataDst = (u_int32_t *)((Levels[0].u8_ptData_Host));

    for(int i=0;i<size;i++)
    {
        u32_ptDataDst[i] = u32_ptDataSrc[i];
    }

    // Fill the other image
    u_int8_t u8_indice = 0;
    for(int i=0;i<LEVELS-1;i++)
    {
        if(u8_indice%2 == 0)
        {
            PyrDown_gpu::run(Levels[i].irow,Levels[i].icol,u8_ptDataIn_Device,u8_ptDataOut_Device,u8_ptTmpBloc_Device);
            checkCudaErrors(cudaMemcpy(Levels[i+1].u8_ptData_Host  , u8_ptDataOut_Device, Levels[i+1].icol*Levels[i+1].irow * sizeof(u_int8_t), cudaMemcpyDeviceToHost));

            int iNbKeyFrames    = Fast_gpu::run_calcKeypoints(u8_ptDataIn_Device,Levels[i].icol,Levels[i].irow, kpLoc_Device, Levels[i].icol*Levels[i].irow*__RATIO__, vCorners_Device, i_tabThreshold[i]);
            int iNbKeyFramesMax = Fast_gpu::run_nonmaxSuppression_gpu(kpLoc_Device, iNbKeyFrames, vCorners_Device, Levels[i].irow, Levels[i].icol, kpLocMax_Device, vCornersMax_Device);
            Levels[i].iNbKeypoints = iNbKeyFrames;
            Levels[i].iNbKeypointsMax = iNbKeyFramesMax;
            //checkCudaErrors(cudaMemcpy(Levels[i].kpLoc_Device  , kpLoc_Device, iNbKeyFrames * sizeof(short2), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(Levels[i].kpLocMax_Device  , kpLocMax_Device, iNbKeyFramesMax * sizeof(short2), cudaMemcpyDeviceToHost));

        }else{
            PyrDown_gpu::run(Levels[i].irow,Levels[i].icol,u8_ptDataOut_Device,u8_ptDataIn_Device,u8_ptTmpBloc_Device);
            checkCudaErrors(cudaMemcpy(Levels[i+1].u8_ptData_Host  , u8_ptDataIn_Device, Levels[i+1].icol*Levels[i+1].irow * sizeof(u_int8_t), cudaMemcpyDeviceToHost));

            int iNbKeyFrames    = Fast_gpu::run_calcKeypoints(u8_ptDataOut_Device,Levels[i].icol,Levels[i].irow, kpLoc_Device, Levels[i].icol*Levels[i].irow*__RATIO__, vCorners_Device, i_tabThreshold[i]);
            int iNbKeyFramesMax = Fast_gpu::run_nonmaxSuppression_gpu(kpLoc_Device, iNbKeyFrames, vCorners_Device, Levels[i].irow, Levels[i].icol, kpLocMax_Device, vCornersMax_Device);
            Levels[i].iNbKeypoints = iNbKeyFrames;
            Levels[i].iNbKeypointsMax = iNbKeyFramesMax;
            //checkCudaErrors(cudaMemcpy(Levels[i].kpLoc_Device  , kpLoc_Device, iNbKeyFrames * sizeof(short2), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(Levels[i].kpLocMax_Device  , kpLocMax_Device, iNbKeyFramesMax * sizeof(short2), cudaMemcpyDeviceToHost));
        }
        u8_indice++;
    }

    // Process the last level f
    int iNbKeyFrames    = Fast_gpu::run_calcKeypoints(u8_ptDataOut_Device,Levels[LEVELS-1].icol,Levels[LEVELS-1].irow, kpLoc_Device, Levels[LEVELS-1].icol*Levels[LEVELS-1].irow*__RATIO__, vCorners_Device, i_tabThreshold[LEVELS-1]);
    int iNbKeyFramesMax = Fast_gpu::run_nonmaxSuppression_gpu(kpLoc_Device, iNbKeyFrames, vCorners_Device, Levels[LEVELS-1].irow, Levels[LEVELS-1].icol, kpLocMax_Device, vCornersMax_Device);
    Levels[LEVELS-1].iNbKeypoints = iNbKeyFrames;
    Levels[LEVELS-1].iNbKeypointsMax = iNbKeyFramesMax;
    //checkCudaErrors(cudaMemcpy(Levels[i].kpLoc_Device  , kpLoc_Device, iNbKeyFrames * sizeof(short2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(Levels[LEVELS-1].kpLocMax_Device  , kpLocMax_Device, iNbKeyFramesMax * sizeof(short2), cudaMemcpyDeviceToHost));

}


////////////////////////////////////
///// \brief KeyFrame::MakePyramidLight
///// \param pt
/////
//void KeyFrame::MakePyramid(u_int8_t *ptData)
//{
//    ///////////////////////////////////
//    // Fill each level with the data

//    //copy the first image
//    checkCudaErrors(cudaMemcpy(Levels[0].ptData  , ptData, row * col * sizeof(u_int8_t), cudaMemcpyHostToDevice));



//    // Fill the other image
//    for(int i=0;i<3;i++)
//    {
//            PyrDown_gpu::run(Levels[i].irow,Levels[i].icol,Levels[i].ptData,Levels[i+1].ptData,ptTmpBloc);
//    }
//}



