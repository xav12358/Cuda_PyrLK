#include "fast.h"
#include "keyframe.h"
#include "pyrdown.h"

#define __RATIO__ 0.05

void Level::BuildLevel(int rows,int cols)
{

    icol = cols;
    irow = rows;

    checkCudaErrors(cudaMalloc((void **)&kpLoc,  icol*irow*__RATIO__ * sizeof(short2)));
    //checkCudaErrors(cudaMalloc((void **)&vCorners,  icol*irow*__RATIO__ * sizeof(int)));
    //checkCudaErrors(cudaMalloc((void **)&ptData,  rows*cols * sizeof(u_int8_t)));

    //checkCudaErrors(cudaMalloc((void **)&kpLocMax,  icol*irow*__RATIO__ * sizeof(short2)));
    //checkCudaErrors(cudaMalloc((void **)&vCornersMax,  icol*irow*__RATIO__ * sizeof(float)));


    u8_ptData = (u_int8_t*)malloc(irow*icol*sizeof(u_int8_t));

    //kpLoc = (short2*)malloc(irow*icol*__RATIO__*sizeof(short2));
    kpLocMax = (short2*)malloc(irow*icol*__RATIO__*sizeof(short2));
    scoreMax = (float*)malloc(irow*icol*__RATIO__*sizeof(float));

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
        checkCudaErrors(cudaMalloc((void **)&u8_ptDataIn,  row*col * sizeof(u_int8_t)));
        checkCudaErrors(cudaMalloc((void **)&u8_ptDataOut,  row*col * sizeof(u_int8_t)));
        checkCudaErrors(cudaMalloc((void **)&u8_ptTmpBloc,  row*col * sizeof(u_int8_t)));

        checkCudaErrors(cudaMalloc((void **)&kpLoc,  icol*irow*__RATIO__ * sizeof(short2)));
        checkCudaErrors(cudaMalloc((void **)&vCorners,  icol*irow*__RATIO__ * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&kpLocMax,  icol*irow*__RATIO__ * sizeof(short2)));
        checkCudaErrors(cudaMalloc((void **)&vCornersMax,  icol*irow*__RATIO__ * sizeof(float)));
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


void KeyFrame::MakePyramid(u_int8_t *u8_ptDataSrc)
{
    ///////////////////////////////////
    // Fill each level with the data

    int i_tabThreshold[4];
    i_tabThreshold[0]= 50;
    i_tabThreshold[1]= 40;
    i_tabThreshold[2]= 30;
    i_tabThreshold[3]= 20;
    //copy the first image
    checkCudaErrors(cudaMemcpy(u8_ptDataIn  , u8_ptDataSrc, irow * icol * sizeof(u_int8_t), cudaMemcpyHostToDevice));

    //Copy the source image to the first level
    int size = irow * icol/4;
    u_int32_t *u32_ptDataSrc = (u_int32_t *)(u8_ptDataSrc);
    u_int32_t *u32_ptDataDst = (u_int32_t *)((Levels[0].u8_ptData));

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
            PyrDown_gpu::run(Levels[i].irow,Levels[i].icol,u8_ptDataIn,u8_ptDataOut,u8_ptTmpBloc);
            checkCudaErrors(cudaMemcpy(Levels[i+1].u8_ptData  , u8_ptDataOut, Levels[i+1].icol*Levels[i+1].irow * sizeof(u_int8_t), cudaMemcpyDeviceToHost));

            int iNbKeyFrames    = Fast_gpu::run_calcKeypoints(u8_ptDataIn,Levels[i].icol,Levels[i].irow, kpLoc, Levels[i].icol*Levels[i].irow*__RATIO__, vCorners, i_tabThreshold[i]);
            int iNbKeyFramesMax = Fast_gpu::run_nonmaxSuppression_gpu(kpLoc, iNbKeyFrames, vCorners, Levels[i].irow, Levels[i].icol, kpLocMax, vCornersMax);
            Levels[i].iNbKeypoints = iNbKeyFrames;
            Levels[i].iNbKeypointsMax = iNbKeyFramesMax;
            //checkCudaErrors(cudaMemcpy(Levels[i].kpLoc  , kpLoc, iNbKeyFrames * sizeof(short2), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(Levels[i].kpLocMax  , kpLocMax, iNbKeyFramesMax * sizeof(short2), cudaMemcpyDeviceToHost));

        }else{
            PyrDown_gpu::run(Levels[i].irow,Levels[i].icol,u8_ptDataOut,u8_ptDataIn,u8_ptTmpBloc);
            checkCudaErrors(cudaMemcpy(Levels[i+1].u8_ptData  , u8_ptDataIn, Levels[i+1].icol*Levels[i+1].irow * sizeof(u_int8_t), cudaMemcpyDeviceToHost));

            int iNbKeyFrames    = Fast_gpu::run_calcKeypoints(u8_ptDataOut,Levels[i].icol,Levels[i].irow, kpLoc, Levels[i].icol*Levels[i].irow*__RATIO__, vCorners, i_tabThreshold[i]);
            int iNbKeyFramesMax = Fast_gpu::run_nonmaxSuppression_gpu(kpLoc, iNbKeyFrames, vCorners, Levels[i].irow, Levels[i].icol, kpLocMax, vCornersMax);
            Levels[i].iNbKeypoints = iNbKeyFrames;
            Levels[i].iNbKeypointsMax = iNbKeyFramesMax;
            //checkCudaErrors(cudaMemcpy(Levels[i].kpLoc  , kpLoc, iNbKeyFrames * sizeof(short2), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(Levels[i].kpLocMax  , kpLocMax, iNbKeyFramesMax * sizeof(short2), cudaMemcpyDeviceToHost));
        }
        u8_indice++;
    }

    // Process the last level f
    int iNbKeyFrames    = Fast_gpu::run_calcKeypoints(u8_ptDataOut,Levels[LEVELS-1].icol,Levels[LEVELS-1].irow, kpLoc, Levels[LEVELS-1].icol*Levels[LEVELS-1].irow*__RATIO__, vCorners, i_tabThreshold[LEVELS-1]);
    int iNbKeyFramesMax = Fast_gpu::run_nonmaxSuppression_gpu(kpLoc, iNbKeyFrames, vCorners, Levels[LEVELS-1].irow, Levels[LEVELS-1].icol, kpLocMax, vCornersMax);
    Levels[LEVELS-1].iNbKeypoints = iNbKeyFrames;
    Levels[LEVELS-1].iNbKeypointsMax = iNbKeyFramesMax;
    //checkCudaErrors(cudaMemcpy(Levels[i].kpLoc  , kpLoc, iNbKeyFrames * sizeof(short2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(Levels[LEVELS-1].kpLocMax  , kpLocMax, iNbKeyFramesMax * sizeof(short2), cudaMemcpyDeviceToHost));

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



