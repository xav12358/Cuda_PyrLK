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

        ptGrayOut[ixo+iyo*w] = min(output,255);
    }

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



/////////////////////////////
/// \brief PyrDown::PyrDown
/// \param rows
/// \param cols
///
PyrDown_gpu::PyrDown_gpu(int rows,int cols)
{
    checkCudaErrors(cudaMalloc((void **)&ptImageL0_Device,  rows * cols * sizeof(u_int8_t)));
    checkCudaErrors(cudaMalloc((void **)&ptImageTmp_Device, rows * cols * sizeof(u_int8_t)));
    checkCudaErrors(cudaMalloc((void **)&ptImageL1_Device,  rows * cols * sizeof(u_int8_t)/4));
    checkCudaErrors(cudaMalloc((void **)&ptImageL2_Device,  rows * cols * sizeof(u_int8_t)/16));
//    checkCudaErrors(cudaMalloc((void **)&ptImageL3_Device,  rows * cols * sizeof(u_int8_t)/64));
}

///////////////////////////////////
/// \brief PyrDown::~PyrDown
///
PyrDown_gpu::~PyrDown_gpu()
{
    // free device memory
    checkCudaErrors(cudaFree(ptImageL0_Device));
    checkCudaErrors(cudaFree(ptImageTmp_Device));
    checkCudaErrors(cudaFree(ptImageL1_Device));
    checkCudaErrors(cudaFree(ptImageL2_Device));
    checkCudaErrors(cudaFree(ptImageL3_Device));
}


///////////////////////////////////
/// \brief PyrDown::run
/// \param rows
/// \param cols
/// \param ptSrc
///
void PyrDown_gpu::run(int rows,int cols,u_int8_t *ptSrc)
{
    checkCudaErrors(cudaMemcpy(ptImageL0_Device , ptSrc, rows * cols * sizeof(u_int8_t), cudaMemcpyHostToDevice));

    dim3 blocks_x(ceil((float)cols / ( BLOCK_SIZE_X)), ceil((float)rows / BLOCK_SIZE_Y));
    dim3 threads_x(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_x_g<<<blocks_x,threads_x>>>(ptImageL0_Device,ptImageTmp_Device, cols,rows);

    dim3 blocks_y(ceil((float)cols / ( BLOCK_SIZE_X)/2.0),ceil((float) rows / BLOCK_SIZE_Y/2.0));
    dim3 threads_y(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_y_g<<<blocks_y,threads_y>>>(ptImageTmp_Device,ptImageL1_Device,  cols/2, rows/2);

//    cv::Mat Image1(rows/2,cols/2,CV_8U);
//    checkCudaErrors(cudaMemcpy(Image1.data , ptImageL1_Device, rows * cols * sizeof(u_int8_t)/4, cudaMemcpyDeviceToHost));
//    cv::imshow("Image L1",Image1);
//    cv::waitKey(-1);


    dim3 blocks_x_L1(ceil(cols / ( BLOCK_SIZE_X/2.0)), ceil(rows / BLOCK_SIZE_Y/2.0));
    dim3 threads_x_L1(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_x_g<<<blocks_x_L1,threads_x_L1>>>(ptImageL1_Device,ptImageTmp_Device,  cols/2, rows/2);

    dim3 blocks_y_L1(ceil(cols /  BLOCK_SIZE_X/4.0), ceil(rows / BLOCK_SIZE_Y/4.0));
    dim3 threads_y_L1(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_y_g<<<blocks_y_L1,threads_y_L1>>>(ptImageTmp_Device,ptImageL2_Device,  cols/4, rows/4);


//    cv::Mat Image2(rows/4,cols/4,CV_8U);
//    checkCudaErrors(cudaMemcpy(Image2.data , ptImageL2_Device, rows * cols * sizeof(u_int8_t)/16, cudaMemcpyDeviceToHost));
//    cv::imshow("Image L2",Image2);
//    cv::waitKey(-1);


//    dim3 blocks_x_L2(ceil(cols /  BLOCK_SIZE_X /4.0), ceil(rows / BLOCK_SIZE_Y/4.0));
//    dim3 threads_x_L2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//    PyrDown_x_g<<<blocks_x_L2,threads_x_L2>>>(ptImageL2_Device,ptImageTmp_Device,  cols/4, rows/4);

//    dim3 blocks_y_L2(ceil(cols /  BLOCK_SIZE_X/8.0), ceil(rows / BLOCK_SIZE_Y/8.0));
//    dim3 threads_y_L2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//    PyrDown_y_g<<<blocks_y_L2,threads_y_L2>>>(ptImageTmp_Device,ptImageL3_Device,  cols/8, rows/8);

//    cv::Mat Image3(rows/8,cols/8,CV_8U);
//    checkCudaErrors(cudaMemcpy(Image3.data , ptImageL3_Device, rows * cols * sizeof(u_int8_t)/64, cudaMemcpyDeviceToHost));
//    cv::imshow("Image L3",Image3);
//    cv::waitKey(-1);

}


void PyrDown_gpu::run(int rows, int cols, u_int8_t *ptIn, u_int8_t *ptOut, u_int8_t *ptTmp)
{


    dim3 blocks_x(ceil((float)cols / ( BLOCK_SIZE_X)), ceil((float)rows / BLOCK_SIZE_Y));
    dim3 threads_x(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_x_g<<<blocks_x,threads_x>>>(ptIn,ptTmp, cols,rows);

    dim3 blocks_y(ceil((float)cols / ( BLOCK_SIZE_X)/2.0),ceil( (float)rows / BLOCK_SIZE_Y/2.0));
    dim3 threads_y(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    PyrDown_y_g<<<blocks_y,threads_y>>>(ptTmp,ptOut,  cols/2, rows/2);



    cv::Mat Image2(rows/2,cols/2,CV_8U);
    checkCudaErrors( cudaMemcpy(Image2.data, ptOut ,rows*cols/4*sizeof(u_int8_t),  cudaMemcpyDeviceToHost) );

//    //cv::Mat Image2(ptKeyFrame->Levels[i].irow,ptKeyFrame->Levels[i].icol,CV_8U,ptData);
//    imshow("Resultat Level IN",Image2);
//    cv::waitKey(-1);

}
