#include "pyrdown.h"
#include <math.h>


#include <cv.h>
#include <opencv2/highgui/highgui.hpp>



//////////////////////////////////////
/// \brief PyrDown_x_g
/// \param ptGrayIn
/// \param ptGrayOut
/// \param w
/// \param h
///
__global__ void PyrDown_x_g(u_int8_t *ptGrayIn,u_int8_t *ptGrayOut,  int w, int h)
{
    __shared__ float  LocalBlock[(BLOCK_SIZE_X+4)*(BLOCK_SIZE_Y)];

    int x = blockIdx.x*blockDim.x;
    int y = blockIdx.y*blockDim.y;

    for(int i=threadIdx.x;i<BLOCK_SIZE_X+4 && (x+i)<w;i=i+blockDim.x)
    {
        for(int j=threadIdx.y;j<BLOCK_SIZE_Y && (y+j)<h;j=j+blockDim.y)
        {
            LocalBlock[i+j*(BLOCK_SIZE_X+4)] = ptGrayIn[x+i + (y+j)*w];
        }
    }

    syncthreads();

    int ix = threadIdx.x+2;
    int iy = threadIdx.y;
    int ixo = blockIdx.x*blockDim.x + threadIdx.x;
    int iyo = blockIdx.y*blockDim.y + threadIdx.y;

    if(ixo<w && iyo<h )
    {
        float p_2   = (float)LocalBlock[ix-2+(iy)*(BLOCK_SIZE_X+4)]/16.0f;
        float p_1   = (float)LocalBlock[ix-1+(iy)*(BLOCK_SIZE_X+4)]/4.0f;
        float p0    = 3.0f*(float)LocalBlock[ix+iy*(BLOCK_SIZE_X+4)]/8.0f;
        float pp1   = (float)LocalBlock[ix+1+(iy)*(BLOCK_SIZE_X+4)]/4.0f;
        float pp2   = (float)LocalBlock[ix+2+(iy)*(BLOCK_SIZE_X+4)]/16.0f;
        int output  = p_2 + p_1 + p0 + pp1 + pp2;

        ptGrayOut[ixo+iyo*w] = min(output,255);//LocalBlock[ix+iy*(BLOCK_SIZE_X+4)];//min(output,255);
    }


    //    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    //    int iy = blockIdx.y*blockDim.y + threadIdx.y;

    //    if(ix<w && iy<h)// && x>2)
    //    {
    //        float p_2   = ptGrayIn[ix-1+(iy)*w]/16.0f;
    //        float p_1   = ptGrayIn[ix-2+(iy)*w]/4.0f;
    //        float p0    = 3.0f*ptGrayIn[ix+iy*w]/8.0f;
    //        float pp1   = ptGrayIn[ix+1+(iy)*w]/4.0f;
    //        float pp2   = ptGrayIn[ix+2+(iy)*w]/16.0f;
    //        int output  = p_2 + p_1 + p0 + pp1 + pp2;
    //        ptGrayOut[ix+iy*w] = min(output,255);
    //    }
}

/////////////////////////////////
/// \brief PyrDown_y_g
/// \param ptGrayIn
/// \param ptGrayOut
/// \param w
/// \param h
///
__global__ void PyrDown_y_g(u_int8_t *ptGrayIn,u_int8_t *ptGrayOut,  int  w, int h)
{
    //    __shared__ unsigned char LocalBlock[(BLOCK_SIZE_X)*(BLOCK_SIZE_Y+4)];
    //int x = blockIdx.x*blockDim.x + threadIdx.x;
    //int y = blockIdx.y*blockDim.y + threadIdx.y;
    //    for(int i = 0; i < BLOCK_SIZE_X && (x + i)*2 < w; i=i+threadDim.x)
    //    {
    //        for(int j=0;j<BLOCK_SIZE_Y+4 && (y+j)*2<h;j=j+threadDim.y)
    //        {
    //            LocalBlock[i+j*(BLOCK_SIZE_Y+4)] = ptGrayIn[(x+i) + 2*(y+j)*w];
    //        }
    //    }
    //    syncthreads();

    //    int ix = threadIdx.x;
    //    int iy = threadIdx.y;

    //    if(ix<w && iy<h-2 && y>2)
    //    {
    //        float p_2   = LocalBlock[ix+(iy-2)*w]/16.0f;
    //        float p_1   = LocalBlock[ix+(iy-1)*w]/4.0f;
    //        float p0    = 3.0f*LocalBlock[ix+iy*w]/8.0f;
    //        float pp1   = LocalBlock[ix+(iy+1)*w]/4.0f;
    //        float pp2   = LocalBlock[ix+(iy+2)*w]/16.0f;

    //        int output  = p_2 + p_1 + p0 + pp1 + pp2;
    //        ptGrayOut[ix+(iy)*w] = min(output,255);
    //    }

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;

    if(ix<w && iy<h)// && y>2)
    {
        float p_2   = ptGrayIn[ix*2+(iy*2-2)*w*2]/16.0f;
        float p_1   = ptGrayIn[ix*2+(iy*2-1)*w*2]/4.0f;
        float p0    = 3.0f*ptGrayIn[ix*2+iy*2*w*2]/8.0f;
        float pp1   = ptGrayIn[ix*2+(iy*2+1)*w*2]/4.0f;
        float pp2   = ptGrayIn[ix*2+(iy*2+2)*w*2]/16.0f;

        int output  = p_2 + p_1 + p0 + pp1 + pp2;
        ptGrayOut[ix+iy*w] = min(output,255);
    }
}


//__global__ void PyrDown(u_int8_t *ptGrayIn,u_int8_t *ptGrayOut,  u_int8_t w, u_int8_t h)
//{
//    __shared__ unsigned char LocalBlock[(BLOCK_SIZE_X*2+4)*(BLOCK_SIZE_Y*2+4)];

//    int x = blockIdx.x*blockDim.x + threadIdx.x;
//    int y = blockIdx.y*blockDim.y + threadIdx.y;
//    for(int i=0;i<BLOCK_SIZE_X+4 && (x+i)<w;i=i+threadDim.x)
//    {
//        for(int j=0;j<BLOCK_SIZE_Y+4 && (y+j)<h;j=j+threadDim.y)
//        {
//            LocalBlock[i+j*BLOCK_SIZE_Y] = ptGrayIn[x+i + (y+j)*w];
//        }
//    }

//    syncthreads();
//    //    int ix = blockIdx.x*blockDim.x + threadIdx.x;
//    //    int iy = blockIdx.y*blockDim.y + threadIdx.y+2;

//    int ix = threadIdx.x+2;
//    int iy = threadIdx.y;


//    if(ix<w && iy<h)
//    {
//        //        float p_2   = ptGrayIn[ix+(iy-2)*w]/16.0f;
//        //        float p_1   = ptGrayIn[ix+(iy-1)*w]/4.0f;
//        //        float p0    = 3.0f*ptGrayIn[ix+iy*w]/8.0f;
//        //        float pp1   = ptGrayIn[ix+(iy+1)*w]/4.0f;
//        //        float pp2   = ptGrayIn[ix+(iy+2)*w]/16.0f;
//        //        int output  = p_2 + p_1 + p0 + pp1 + pp2;

//        float p_2   = LocalBlock[ix-2+(iy)*w]/16.0f;
//        float p_1   = LocalBlock[ix-1+(iy)*w]/4.0f;
//        float p0    = 3.0f*LocalBlock[ix+iy*w]/8.0f;
//        float pp1   = LocalBlock[ix+1+(iy)*w]/4.0f;
//        float pp2   = LocalBlock[ix+2+(iy)*w]/16.0f;
//        int output  = p_2 + p_1 + p0 + pp1 + pp2;

//        ptGrayOut[ix+iy*w] = min(output,255);
//    }
//}



/////////////////////////////
/// \brief PyrDown::PyrDown
/// \param rows
/// \param cols
///
PyrDown_gpu::PyrDown_gpu(int rows,int cols)
{
    checkCudaErrors(cudaMalloc((void **)&ptImageL0,  rows * cols * sizeof(u_int8_t)));
    checkCudaErrors(cudaMalloc((void **)&ptImageTmp, rows * cols * sizeof(u_int8_t)));
    checkCudaErrors(cudaMalloc((void **)&ptImageL1,  rows * cols * sizeof(u_int8_t)/4));
    checkCudaErrors(cudaMalloc((void **)&ptImageL2,  rows * cols * sizeof(u_int8_t)/16));
    checkCudaErrors(cudaMalloc((void **)&ptImageL3,  rows * cols * sizeof(u_int8_t)/64));
}

///////////////////////////////////
/// \brief PyrDown::~PyrDown
///
PyrDown_gpu::~PyrDown_gpu()
{
    // free device memory
    checkCudaErrors(cudaFree(ptImageL0));
    checkCudaErrors(cudaFree(ptImageTmp));
    checkCudaErrors(cudaFree(ptImageL1));
    checkCudaErrors(cudaFree(ptImageL2));
    checkCudaErrors(cudaFree(ptImageL3));
}


///////////////////////////////////
/// \brief PyrDown::run
/// \param rows
/// \param cols
/// \param ptSrc
///
void PyrDown_gpu::run(int rows,int cols,u_int8_t *ptSrc)
{
    checkCudaErrors(cudaMemcpy(ptImageL0 , ptSrc, rows * cols * sizeof(u_int8_t), cudaMemcpyHostToDevice));

    dim3 blocks_x(ceil(cols / ( BLOCK_SIZE_X)), ceil(rows / BLOCK_SIZE_Y));
    dim3 threads_x(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_x_g<<<blocks_x,threads_x>>>(ptImageL0,ptImageTmp, cols,rows);

    dim3 blocks_y(ceil(cols / ( BLOCK_SIZE_X)/2.0),ceil( rows / BLOCK_SIZE_Y/2.0));
    dim3 threads_y(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_y_g<<<blocks_y,threads_y>>>(ptImageTmp,ptImageL1,  cols/2, rows/2);




    dim3 blocks_x_L1(ceil(cols / ( BLOCK_SIZE_X/2.0)), ceil(rows / BLOCK_SIZE_Y/2.0));
    dim3 threads_x_L1(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_x_g<<<blocks_x_L1,threads_x_L1>>>(ptImageL1,ptImageTmp,  cols/2, rows/2);

    dim3 blocks_y_L1(ceil(cols /  BLOCK_SIZE_X/4.0), ceil(rows / BLOCK_SIZE_Y/4.0));
    dim3 threads_y_L1(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_y_g<<<blocks_y_L1,threads_y_L1>>>(ptImageTmp,ptImageL2,  cols/4, rows/4);




    dim3 blocks_x_L2(ceil(cols /  BLOCK_SIZE_X /4.0), ceil(rows / BLOCK_SIZE_Y/4.0));
    dim3 threads_x_L2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_x_g<<<blocks_x_L2,threads_x_L2>>>(ptImageL2,ptImageTmp,  cols/4, rows/4);

    dim3 blocks_y_L2(ceil(cols /  BLOCK_SIZE_X/8.0), ceil(rows / BLOCK_SIZE_Y/8.0));
    dim3 threads_y_L2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_y_g<<<blocks_y_L2,threads_y_L2>>>(ptImageTmp,ptImageL3,  cols/8, rows/8);

}

/////////////////////////////////////
/// \brief PyrDown::run
/// \param rows
/// \param cols
/// \param ptSrc
/// \param ilevel
///
void PyrDown_gpu::run(int rows,int cols,u_int8_t *ptSrc,int ilevel)
{

    //    //    switch(ilevel)
    //    //    {
    //    //    case 1:
    //    dim3 blocks_x(cols / ( BLOCK_SIZE_X), rows / BLOCK_SIZE_Y);
    //    dim3 threads_x(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    //    PyrDown_x_g<<<blocks_x,threads_x>>>(ptImageL1,ptImageTmp,  cols, rows);

    //    dim3 blocks_y(cols/2 / ( BLOCK_SIZE_X), rows/2 / BLOCK_SIZE_Y);
    //    dim3 threads_y(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    //    PyrDown_y_g<<<blocks_y,threads_y>>>(ptImageTmp,ptImageL2,  cols/2, rows/2);
    //    //        break;
    //    /*
    //    case 2:
    //        dim3 blocks_x(cols / ( BLOCK_SIZE_X), rows / BLOCK_SIZE_Y);
    //        dim3 threads_x(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    //        PyrDown_x_g<<<blocks_x,threads_x>>>(ptImageL2,ptImageTmp,  cols, rows);

    //        dim3 blocks_y(cols/2 / ( BLOCK_SIZE_X), rows/2 / BLOCK_SIZE_Y);
    //        dim3 threads_y(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    //        PyrDown_y_g<<<blocks_y,threads_y>>>(ptImageTmp,ptImageL3,  cols/2, rows/2);
    //        break*/;
    //    //    }


}
