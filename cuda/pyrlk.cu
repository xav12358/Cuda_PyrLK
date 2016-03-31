#include "pyrlk.h"

#include "global_var.h"
#include "patchtracker.h"
#include "pyrdown.h"

#include <cv.h>
#include <opencv2/highgui/highgui.hpp>

//#define NB_FEATURE_MAX                          120
//#define THRESHOLD                               0.01


texture<unsigned char, 2,cudaReadModeElementType> Image_I;
texture<unsigned char, 2,cudaReadModeElementType> Image_J;


__global__ void lkflow_kernel(
        float2 *prevPt,
        float2 *nextPt,
        u_int8_t *status,
        int rows,
        int cols,
        int iiter,
        int ilevel)
{
    // declare some shared memory
    __shared__ float smem[HALF_WIN*2+3][HALF_WIN*2+3] ;
    __shared__ float smemIy[HALF_WIN*2+1][HALF_WIN*2+1] ;
    __shared__ float smemIx[HALF_WIN*2+1][HALF_WIN*2+1] ;
    __shared__ float smemA11[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float smemA21[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float smemA22[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float smemb1[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float smemb2[BLOCK_SIZE_Y][BLOCK_SIZE_X];


    for(int y = threadIdx.y ;y< BLOCK_SIZE_Y; y = y+blockDim.y)
    {
        for(int x = threadIdx.x ;x< BLOCK_SIZE_X; x = x+blockDim.x)
        {
            smemA11[y][x] = 0;
            smemA21[y][x] = 0;
            smemA22[y][x] = 0;
            smemb1[y][x] = 0;
            smemb2[y][x] = 0;
        }
    }


    // copy into local memory
    __shared__ float2 PrevPtL;
    if (threadIdx.x ==0 && threadIdx.y == 0)
    {
        PrevPtL.x=  prevPt[blockIdx.x].x*1.0/(1<<(ilevel))- (float)(HALF_WIN);
        PrevPtL.y=  prevPt[blockIdx.x].y*1.0/(1<<(ilevel))- (float)(HALF_WIN);
    }

    syncthreads();


    for(int y= threadIdx.y-1 ;y< HALF_WIN*2+3-1;y = y+blockDim.y)
    {
        for(int x= threadIdx.x-1 ;x< HALF_WIN*2+3-1;x = x+blockDim.x)
        {
            float2 CurrentPt;
            CurrentPt.x = PrevPtL.x+x;
            CurrentPt.y = PrevPtL.y+y;
            smem[y+1][x+1] =  tex2D(Image_I,CurrentPt.x,CurrentPt.y);//read_imageui( I, bilinSampler, PrevPtL+(float2)(x,y) ).x;
        }
    }
    syncthreads();


    //////////////////////////////////////////////////////
    // Compute derivative
    int ValY_1X_1;
    int ValY_1X;
    int ValY_1Xp1;

    int ValYX_1;
    int ValYXp1;

    int ValYp1X_1;
    int ValYp1X;
    int ValYp1Xp1;

    for(int y= threadIdx.y+1 ;y<= HALF_WIN*2+1;y = y+blockDim.y)
    {
        for(int x= threadIdx.x+1 ;x<= HALF_WIN*2+1;x = x+blockDim.x)
        {

            ValY_1X_1  = smem[y-1][x-1];
            ValY_1X    = smem[y-1][x];
            ValY_1Xp1  = smem[y-1][x+1];

            ValYX_1    = smem[y][x-1];
            ValYXp1    = smem[y][x+1];

            ValYp1X_1  = smem[y+1][x-1];
            ValYp1X    = smem[y+1][x];
            ValYp1Xp1  = smem[y+1][x+1];

            smemIx[y-1][x-1] = 3.0*( ValY_1Xp1 +  ValYp1Xp1 - ValY_1X_1 -ValYp1X_1 ) + 10.0*(ValYXp1 - ValYX_1);
            smemIy[y-1][x-1] = 3.0*( ValYp1X_1 +  ValYp1Xp1 - ValY_1X_1 -ValY_1Xp1 ) + 10.0*(ValYp1X - ValY_1X);

        }
    }
    syncthreads();


    ////////////////////////////////////////
    // Calculated A (only on one thread)
    __shared__ float A11,A12,A22;

    if (threadIdx.x ==0 && threadIdx.y == 0)
    {
        A11 = 0;
        A12 = 0;
        A22 = 0;
    }
    syncthreads();

    for(int y= threadIdx.y ;y< HALF_WIN*2+1;y = y+blockDim.y)
    {
        for(int x= threadIdx.x ;x< HALF_WIN*2+1;x = x+blockDim.x)
        {
            smemA11[threadIdx.y][threadIdx.x] +=smemIx[y][x]*smemIx[y][x];
            smemA21[threadIdx.y][threadIdx.x] +=smemIx[y][x]*smemIy[y][x];
            smemA22[threadIdx.y][threadIdx.x] +=smemIy[y][x]*smemIy[y][x];
        }
    }

    syncthreads();

    if (threadIdx.x ==0 && threadIdx.y == 0)
    {
        for(int y= 0 ;y< blockDim.y;y++)
        {
            for(int x= 0 ;x< blockDim.x;x++)
            {
                A11 +=smemA11[y][x];
                A12 +=smemA21[y][x];
                A22 +=smemA22[y][x];
            }
        }

        float D = A11 * A22 - A12 * A12;

        //        if (D < 1.192092896e-07f)
        //        {
        //            if (tid == 0 && level == 0)
        //                status[gid] = 0;

        //            return;
        //        }

        A11 /= D;
        A12 /= D;
        A22 /= D;

    }

    syncthreads();

    /////////////////////////////////////////
    // Compute optical flow
    __shared__ float2 NextPtL;

    if (threadIdx.x ==0 && threadIdx.y == 0)
    {
        NextPtL.x = nextPt[blockIdx.x].x*2.0 - HALF_WIN;
        NextPtL.y = nextPt[blockIdx.x].y*2.0 - HALF_WIN;
    }

    syncthreads();

    for (int k = 0; k < iiter; k++)
    {
        if (NextPtL.x < -HALF_WIN || NextPtL.x >= cols || NextPtL.y < -HALF_WIN || NextPtL.y >= rows)
        {
            status[blockIdx.x] = 0;
            return;
        }

        float b1 = 0;
        float b2 = 0;

        for(int y= threadIdx.y ;y< HALF_WIN*2+1;y = y+blockDim.y)
        {
            for(int x= threadIdx.x ;x< HALF_WIN*2+1;x = x+blockDim.x)
            {
                float2 CurrentPt;
                CurrentPt.x = NextPtL.x+x;
                CurrentPt.y = NextPtL.y+y;

                float I_val = smem[y+1][x+1];
                float J_val = tex2D(Image_J,CurrentPt.x,CurrentPt.y);//GetPixel(J,NextPtL+(float2)(x,y));//read_imageui( J, bilinSampler, NextPtL+(float2)(x,y) ).x;
                float diff = (J_val - I_val);//*32.0;
                smemb1[threadIdx.y][threadIdx.x] +=diff*smemIx[y][x];
                smemb2[threadIdx.y][threadIdx.x] +=diff*smemIy[y][x];
            }
        }


        syncthreads();
        __shared__ float2 delta;
        if (threadIdx.x ==0 && threadIdx.y == 0)
        {
            b1 = 0;
            b2 = 0;
            for(int y= 0 ;y< blockDim.y;y++)
            {
                for(int x= 0 ;x< blockDim.x;x++)
                {
                    b1 +=smemb1[y][x];
                    b2 +=smemb2[y][x];
                }
            }


            delta.x = A12 * b2 - A22 * b1;
            delta.y = A12 * b1 - A11 * b2;

            NextPtL.x += delta.x;
            NextPtL.y += delta.y;
        }

        syncthreads();

        if(fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
            break;
    }


    if (threadIdx.x ==0 && threadIdx.y == 0)
    {

        NextPtL.x += HALF_WIN;
        NextPtL.y += HALF_WIN;

        nextPt[blockIdx.x] = NextPtL;
    }

}



__global__ void lkflowSparse_kernel(u_int8_t *u8_PatchsWithBorder,
                                    float2 *prevPt,
                                    float2 *nextPt,
                                    u_int8_t *status,
                                    int rows,
                                    int cols,
                                    int iiter,
                                    float2 * f2tmp)
{
    // declare some shared memory
    __shared__ float smem[PATCH_SIZE_WITH_BORDER][PATCH_SIZE_WITH_BORDER] ;
    __shared__ float smemIx[PATCH_SIZE_L][PATCH_SIZE_L];
    __shared__ float smemIy[PATCH_SIZE_L][PATCH_SIZE_L];
    __shared__ float smemA11[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float smemA21[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float smemA22[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float smemb1[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float smemb2[BLOCK_SIZE_Y][BLOCK_SIZE_X];


    for(int y = threadIdx.y ;y< BLOCK_SIZE_Y; y = y+blockDim.y)
    {
        for(int x = threadIdx.x ;x< BLOCK_SIZE_X; x = x+blockDim.x)
        {
            smemA11[y][x] = 0;
            smemA21[y][x] = 0;
            smemA22[y][x] = 0;
            smemb1[y][x] = 0;
            smemb2[y][x] = 0;
        }
    }


    syncthreads();

    for(int y= threadIdx.y-1 ;y< PATCH_SIZE_WITH_BORDER-1;y = y+blockDim.y)
    {
        for(int x= threadIdx.x-1 ;x< PATCH_SIZE_WITH_BORDER-1;x = x+blockDim.x)
        {
            /* float2 CurrentPt;
            CurrentPt.x = PrevPtL.x+x;
            CurrentPt.y = PrevPtL.y+y;*/
            int xc = x+1;
            int yc = y+1;
//            f2tmp[xc+yc*PATCH_SIZE_WITH_BORDER].x = blockIdx.x;
//            f2tmp[xc+yc*PATCH_SIZE_WITH_BORDER].y = xc+yc*PATCH_SIZE_WITH_BORDER + blockIdx.x*PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER;
            smem[y+1][x+1] =  u8_PatchsWithBorder[(int)(xc+yc*PATCH_SIZE_WITH_BORDER + blockIdx.x*PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER)];//tex2D(Image_I,CurrentPt.x,CurrentPt.y);//read_imageui( I, bilinSampler, PrevPtL+(float2)(x,y) ).x;
        //f2tmp[xc+yc*PATCH_SIZE_WITH_BORDER].x = smem[y+1][x+1];
        }
    }
    syncthreads();


    //    //////////////////////////////////////////////////////
    // Compute derivative
    int ValY_1X_1;
    int ValY_1X;
    int ValY_1Xp1;

    int ValYX_1;
    int ValYXp1;

    int ValYp1X_1;
    int ValYp1X;
    int ValYp1Xp1;

    for(int y= threadIdx.y+1 ;y<= PATCH_SIZE_L;y = y+blockDim.y)
    {
        for(int x= threadIdx.x+1 ;x<= PATCH_SIZE_L;x = x+blockDim.x)
        {


            ValY_1X_1  = smem[y-1][x-1];
            ValY_1X    = smem[y-1][x];
            ValY_1Xp1  = smem[y-1][x+1];

            ValYX_1    = smem[y][x-1];
            ValYXp1    = smem[y][x+1];

            ValYp1X_1  = smem[y+1][x-1];
            ValYp1X    = smem[y+1][x];
            ValYp1Xp1  = smem[y+1][x+1];

            smemIx[y-1][x-1] = 3.0*( ValY_1Xp1 +  ValYp1Xp1 - ValY_1X_1 -ValYp1X_1 ) + 10.0*(ValYXp1 - ValYX_1);
            smemIy[y-1][x-1] = 3.0*( ValYp1X_1 +  ValYp1Xp1 - ValY_1X_1 -ValY_1Xp1 ) + 10.0*(ValYp1X - ValY_1X);
//            f2tmp[(x-1)+(y-1)*PATCH_SIZE_WITH_BORDER].x = smemIx[y-1][x-1];
//            f2tmp[(x-1)+(y-1)*PATCH_SIZE_WITH_BORDER].y = smemIy[y-1][x-1];

        }
    }
    syncthreads();


    ////////////////////////////////////////
    // Calculated A (only on one thread)
    __shared__ float A11,A12,A22;

    if (threadIdx.x ==0 && threadIdx.y == 0)
    {
        A11 = 0;
        A12 = 0;
        A22 = 0;
    }
    syncthreads();

    for(int y= threadIdx.y ;y< PATCH_SIZE_L;y = y+blockDim.y)
    {
        for(int x= threadIdx.x ;x< PATCH_SIZE_L;x = x+blockDim.x)
        {
            smemA11[threadIdx.y][threadIdx.x] +=smemIx[y][x]*smemIx[y][x];
            smemA21[threadIdx.y][threadIdx.x] +=smemIx[y][x]*smemIy[y][x];
            smemA22[threadIdx.y][threadIdx.x] +=smemIy[y][x]*smemIy[y][x];
            //f2tmp[x+y*PATCH_SIZE_WITH_BORDER].x = smemIy[y][x]*smemIy[y][x];
            //f2tmp[x+y*PATCH_SIZE_WITH_BORDER].y = 10.3;//smemIx[y][x]*smemIy[y][x];
        }
    }

    syncthreads();

    if (threadIdx.x ==0 && threadIdx.y == 0)
    {
        for(int y= 0 ;y< blockDim.y;y++)
        {
            for(int x= 0 ;x< blockDim.x;x++)
            {
                A11 +=smemA11[y][x];
                A12 +=smemA21[y][x];
                A22 +=smemA22[y][x];
            }
        }

        float D = A11 * A22 - A12 * A12;

        //        if (D < 1.192092896e-07f)
        //        {
        //            if (tid == 0 && level == 0)
        //                status[gid] = 0;

        //            return;
        //        }

        A11 /= D;
        A12 /= D;
        A22 /= D;

    }

    syncthreads();

    /////////////////////////////////////////
    // Compute optical flow
    __shared__ float2 NextPtL;

    if (threadIdx.x ==0 && threadIdx.y == 0)
    {
        NextPtL.x = nextPt[blockIdx.x].x - HALF_WIN;
        NextPtL.y = nextPt[blockIdx.x].y - HALF_WIN;
    }

    syncthreads();

    for (int k = 0; k < iiter; k++)
    {
        if (NextPtL.x < -HALF_WIN || NextPtL.x >= cols || NextPtL.y < -HALF_WIN || NextPtL.y >= rows)
        {
            status[blockIdx.x] = 0;
            return;
        }

        float b1 = 0;
        float b2 = 0;

        for(int y= threadIdx.y ;y< PATCH_SIZE_L;y = y+blockDim.y)
        {
            for(int x= threadIdx.x ;x< PATCH_SIZE_L;x = x+blockDim.x)
            {
                float2 CurrentPt;
                CurrentPt.x = NextPtL.x+x;
                CurrentPt.y = NextPtL.y+y;

                float I_val = smem[y+1][x+1];
                float J_val = tex2D(Image_J,CurrentPt.x,CurrentPt.y);
                float diff = (J_val - I_val);
//                f2tmp[x+y*PATCH_SIZE_WITH_BORDER].x = I_val;//CurrentPt.x;//I_val;
//                f2tmp[x+y*PATCH_SIZE_WITH_BORDER].y = 10.8;//CurrentPt.y;//J_val;

//                f2tmp[x+y*PATCH_SIZE_WITH_BORDER].x = smemIx[y][x];
//                f2tmp[x+y*PATCH_SIZE_WITH_BORDER].x = smemIy[y][x];
                smemb1[threadIdx.y][threadIdx.x] +=diff*smemIx[y][x];
                smemb2[threadIdx.y][threadIdx.x] +=diff*smemIy[y][x];

//                f2tmp[x+y*PATCH_SIZE_WITH_BORDER].x = smemIx[y][x];
//                f2tmp[x+y*PATCH_SIZE_WITH_BORDER].y = -18.2;
            }
        }


        syncthreads();
        __shared__ float2 delta;
        if (threadIdx.x ==0 && threadIdx.y == 0)
        {
            b1 = 0;
            b2 = 0;
            for(int y= 0 ;y< blockDim.y;y++)
            {
                for(int x= 0 ;x< blockDim.x;x++)
                {
                    b1 +=smemb1[y][x];
                    b2 +=smemb2[y][x];

//                    f2tmp[x+y*PATCH_SIZE_WITH_BORDER].x = b1;
//                    f2tmp[x+y*PATCH_SIZE_WITH_BORDER].y = 10.2;//b2;
                }
            }


//            f2tmp[threadIdx.x+threadIdx.y*PATCH_SIZE_WITH_BORDER].x = b2;
//            f2tmp[threadIdx.x+threadIdx.y*PATCH_SIZE_WITH_BORDER].y = -199.2;

            delta.x = A12 * b2 - A22 * b1;
            delta.y = A12 * b1 - A11 * b2;

//            f2tmp[threadIdx.x+threadIdx.y*PATCH_SIZE_WITH_BORDER].x = delta.y;
//            f2tmp[threadIdx.x+threadIdx.y*PATCH_SIZE_WITH_BORDER].y = -199.8;

            NextPtL.x += delta.x;
            NextPtL.y += delta.y;
        }

        syncthreads();

        if(fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
            break;
    }


    if (threadIdx.x ==0 && threadIdx.y == 0)
    {

        NextPtL.x += HALF_WIN;
        NextPtL.y += HALF_WIN;

        nextPt[blockIdx.x] = NextPtL;
    }

}



//////////////////////////////////////
/// \brief PyrLK_gpu::PyrLK_gpu
///
PyrLK_gpu::PyrLK_gpu(int rows, int cols, int iNbMaxFeatures):
    iNbMaxFeaturesLK(iNbMaxFeatures)
{

    ////////////////////////////////////////
    // Create pyrdown for each image
    std::cout << "Compute PyrDown L1" << std::endl;
    ptPyrDownI = new PyrDown_gpu(rows,cols);

    std::cout << "Compute PyrDown L2" << std::endl;
    ptPyrDownJ = new PyrDown_gpu(rows,cols);

    ///////////////////////////////////
    // Create variable in device memory
    checkCudaErrors(cudaMalloc((void **)&f2_PointsPrevDevice,  iNbMaxFeatures * sizeof(float2)));
    checkCudaErrors(cudaMalloc((void **)&f2_PointsNextDevice,  iNbMaxFeatures * sizeof(float2)));
    checkCudaErrors(cudaMalloc((void **)&u8_StatusDevice,  iNbMaxFeatures * sizeof(u_int8_t)));

    //    f2_PointsPrevHost  = (float2*)malloc(iNbMaxFeatures*sizeof(float2));
    f2_PointsNextHost  = (float2*)malloc(iNbMaxFeatures*sizeof(float2));
    u8_StatusHost      = (u_int8_t*)malloc(iNbMaxFeatures*sizeof(u_int8_t));
}




//////////////////////////////////
/// \brief PyrLK_gpu::~PyrLK_gpu
///
PyrLK_gpu::~PyrLK_gpu()
{

}

///////////////////////////////////////////
/// \brief PyrLK_gpu::run_sparse
/// \param u8_ImagePrevHost
/// \param u8_ImageNextHost
/// \param h
/// \param w
/// \param f2_PointsPrevHost
/// \param iNbPoints
///
void PyrLK_gpu::run_sparse(u_int8_t  *u8_ImagePrevHost,u_int8_t *u8_ImageNextHost,int h,int w,float2 *f2_PointsPrevHost,int iNbPointsToSearch)
{

    //////////////////////////////////////
    // Compute PyrDown for the two images
    ptPyrDownI->run(h,w,u8_ImagePrevHost);
    ptPyrDownJ->run(h,w,u8_ImageNextHost);

    ///////////////////////////////////
    // Create initial next points equal to the previous points
    int lvls = 3;
    for(int i=0;i<iNbPointsToSearch;i++)
    {
        f2_PointsNextHost[i].x = (f2_PointsPrevHost[i].x)/(1<<(lvls));
        f2_PointsNextHost[i].y = (f2_PointsPrevHost[i].y)/(1<<(lvls));
    }


    checkCudaErrors( cudaMemcpy(f2_PointsPrevDevice, f2_PointsPrevHost, iNbPointsToSearch*sizeof(float2), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(f2_PointsNextDevice, f2_PointsNextHost, iNbPointsToSearch*sizeof(float2), cudaMemcpyHostToDevice) );



    ////////////////////////////////////////////
    //    float2 *ftmp,*ftmp_CU;
    //    int valstep  = (10*2+3);
    //    checkCudaErrors(cudaMalloc((void **)&ftmp_CU,  valstep*valstep * sizeof(float2)));
    //    ftmp    = (float2*)malloc( valstep*valstep*sizeof(float2));

    ///////////////////////////////////////////
    // Compute Optical flow for each point at each level
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(iNbPointsToSearch, 1);
    for( int i=lvls-1; i>=0 ; i-- )
    {

        //        std::cout << "***************level " << i << " valstep " << valstep << " pow(2,i); " << pow(2,i) << " h/dlevel * w/dlevel "<< h/pow(2,i) << "  " <<w/pow(2,i) <<std::endl;
        int dlevel = pow(2,i);
        switch(i)
        {
        case 0:
            // create texture object
            checkCudaErrors(cudaBindTexture2D(0,&Image_I,ptPyrDownI->ptImageL0,&desc,w/dlevel , h/dlevel,w/dlevel));
            checkCudaErrors(cudaBindTexture2D(0,&Image_J,ptPyrDownJ->ptImageL0,&desc,w/dlevel , h/dlevel,w/dlevel));
            break;

        case 1:
            checkCudaErrors(cudaBindTexture2D(0,&Image_I,ptPyrDownI->ptImageL1,&desc,w/dlevel , h/dlevel,w/dlevel));
            checkCudaErrors(cudaBindTexture2D(0,&Image_J,ptPyrDownJ->ptImageL1,&desc,w/dlevel , h/dlevel,w/dlevel));
            break;

        case 2:
            checkCudaErrors(cudaBindTexture2D(0,&Image_I,ptPyrDownI->ptImageL2,&desc,w/dlevel , h/dlevel,w/dlevel));
            checkCudaErrors(cudaBindTexture2D(0,&Image_J,ptPyrDownJ->ptImageL2,&desc,w/dlevel , h/dlevel,w/dlevel));
            break;
        case 3:
            //dlevel = 8;
            std::cout << "dlevel " << dlevel << std::endl;
            //            checkCudaErrors(cudaBindTexture2D(0,&Image_I,ptPyrDownI->ptImageL3,&desc,w/dlevel , h/dlevel,w/dlevel));
            //            checkCudaErrors(cudaBindTexture2D(0,&Image_J,ptPyrDownJ->ptImageL3,&desc,w/dlevel , h/dlevel,w/dlevel));
            //checkCudaErrors(cudaBindTexture2D(0,&Image_I,ptPyrDownI->ptImageL3,&desc,160 , 120,160));
            //checkCudaErrors(cudaBindTexture2D(0,&Image_J,ptPyrDownJ->ptImageL3,&desc,160 , 120,160));
            break;

        }

        //        lkflow_kernel<<<blocks,threads>>>(PrevPt_CU,NextPt_CU,uStatus_CU,ftmp_CU,h/dlevel,w/dlevel,10,i);
        lkflow_kernel<<<blocks,threads>>>(f2_PointsPrevDevice,f2_PointsNextDevice,u8_StatusDevice,h/dlevel,w/dlevel,13,i);

        checkCudaErrors(cudaUnbindTexture(Image_I));
        checkCudaErrors(cudaUnbindTexture(Image_J));
    }

    checkCudaErrors( cudaMemcpy(f2_PointsNextHost, f2_PointsNextDevice, iNbPointsToSearch*sizeof(float2), cudaMemcpyDeviceToHost) );


    cv::Mat *ImageConcat = new cv::Mat(h, w*2, CV_8U);
    cv::Mat left(*ImageConcat, cv::Rect(0, 0, w, h));
    cv::Mat Image1(h,w,CV_8U,u8_ImagePrevHost);
    Image1.copyTo(left);
    cv::Mat right(*ImageConcat, cv::Rect(w, 0, w,h));
    cv::Mat Image2(h,w,CV_8U,u8_ImageNextHost);
    Image2.copyTo(right);

    cv::Mat im3concat;

    ImageConcat->copyTo(im3concat);

    for(int j=0;j<iNbPointsToSearch;j++)
    {
        cv::line( *ImageConcat, cv::Point( f2_PointsPrevHost[j].x, f2_PointsPrevHost[j].y ), cv::Point( f2_PointsNextHost[j].x+640, f2_PointsNextHost[j].y ) ,cv::Scalar(0,0,0));
        std::cout << "prev :  "<< f2_PointsPrevHost[j].x << " " << f2_PointsPrevHost[j].y << " Next " << f2_PointsNextHost[j].x << " " << f2_PointsNextHost[j].y << std::endl;
    }


    cv::imshow("ImageConcat",*ImageConcat);
    cv::waitKey(-1);

}



////////////////////////////////////////////
/// \brief PyrLK_gpu::run_sparsePatch
/// \param u8_ImagePrevHost
/// \param u8_PatchWithBorderDevice
/// \param h
/// \param w
/// \param f2_PointsPrevHost
/// \param iNbPoints
///
void PyrLK_gpu::run_sparsePatch(u_int8_t  *u8_ImagePrevDevice, PatchTracker *ptTracker, int h, int w)
{

    static int iter = 10;
    ///////////////////////////////////
    // Create initial next points equal to the previous points
    int lvls = 3;
    for(int i=0;i<ptTracker->i_IndiceFeaturesToWarp;i++)
    {
        f2_PointsNextHost[i].x = (ptTracker->f2_PositionFeaturesHost[i].x);
        f2_PointsNextHost[i].y = (ptTracker->f2_PositionFeaturesHost[i].y);
    }

    checkCudaErrors( cudaMemcpy(f2_PointsPrevDevice, ptTracker->f2_PositionFeaturesHost, ptTracker->i_IndiceFeaturesToWarp*sizeof(float2), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(f2_PointsNextDevice, f2_PointsNextHost, ptTracker->i_IndiceFeaturesToWarp*sizeof(float2), cudaMemcpyHostToDevice) );


    ///////////////////////////////////////////
    // Compute Optical flow for each patch at each level
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(ptTracker->i_IndiceFeaturesToWarp, 1);


    checkCudaErrors(cudaBindTexture2D(0,&Image_J,u8_ImagePrevDevice,&desc,w ,h,w));

    float2 *f2tmpDevice;
    float2 *f2tmpHost  = (float2*)malloc(PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER * sizeof(float2));
    checkCudaErrors(cudaMalloc((void **)&f2tmpDevice,  PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER * sizeof(float2)));

    lkflowSparse_kernel<<<blocks,threads>>>(ptTracker->u8_PatchsWithBorderDevice,
                                            ptTracker->f2_PositionFeaturesDevice,
                                            f2_PointsNextDevice,
                                            u8_StatusDevice,
                                            h,
                                            w,
                                            iter++,
                                            f2tmpDevice);

    /*float2 *ptf2_PointsNextHost  = (float2*)malloc(256*sizeof(float2));
    float2 *ptf2_PointsPrevDevice;
    checkCudaErrors(cudaMalloc((void **)&ptf2_PointsPrevDevice,  256 * sizeof(float2)));
*/

    checkCudaErrors(cudaMemcpy(f2tmpHost,f2tmpDevice, PATCH_SIZE_WITH_BORDER*PATCH_SIZE_WITH_BORDER * sizeof(float2),cudaMemcpyDeviceToHost));

    for(int i=0;i<PATCH_SIZE_WITH_BORDER;i++)
    {
        for(int j=0;j<PATCH_SIZE_WITH_BORDER;j++)
        {

            std::cout << f2tmpHost[i+j*PATCH_SIZE_WITH_BORDER].x << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    for(int i=0;i<PATCH_SIZE_WITH_BORDER;i++)
    {
        for(int j=0;j<PATCH_SIZE_WITH_BORDER;j++)
        {

            std::cout << f2tmpHost[i+j*PATCH_SIZE_WITH_BORDER].y << " ";
        }
        std::cout << std::endl;
    }


    checkCudaErrors(cudaMemcpy(f2_PointsNextHost,f2_PointsNextDevice, ptTracker->i_IndiceFeaturesToWarp*sizeof(float2), cudaMemcpyDeviceToHost) );

    for(int i=0;i<ptTracker->i_IndiceFeaturesToWarp;i++)
    {
        std::cout << "Prev " << ptTracker->f2_PositionFeaturesHost[i].x << "  " << ptTracker->f2_PositionFeaturesHost[i].y << " Next " << f2_PointsNextHost[i].x << "  " << f2_PointsNextHost[i].y << std::endl;
    }
}
