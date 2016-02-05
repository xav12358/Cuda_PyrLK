#ifndef GLOBAL_HH
#define GLOBAL_HH

#pragma once

#define LEVELS 4

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#include <stdio.h>
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

#endif // GLOBAL_HH

