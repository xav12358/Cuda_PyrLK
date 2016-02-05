#ifndef PYRLK_H
#define PYRLK_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>



#include "global_var.h"

class PyrLK_gpu
{

public:


    PyrLK_gpu();
    ~PyrLK_gpu();
    void run_sparse(u_int8_t  *Idata,u_int8_t*Jdata,int h,int w);

};

#endif // PYRLK_H
