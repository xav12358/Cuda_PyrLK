#ifndef PYRDOWN_H
#define PYRDOWN_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32


// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        printf( "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


class PyrDown_gpu
{

public:


    u_int8_t *ptImageTmp;
    u_int8_t *ptImageL0;
    u_int8_t *ptImageL1;
    u_int8_t *ptImageL2;
    u_int8_t *ptImageL3;

    PyrDown_gpu(int rows, int cols);
    ~PyrDown_gpu();
    void run(int rows,int cols,u_int8_t *ptSrc);
    void run(int rows,int cols,u_int8_t *ptSrc,int ilevel);
};

#endif // PYRDOWN_H
