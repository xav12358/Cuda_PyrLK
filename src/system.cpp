#include <include/system.h>
#include "/home/lineo/opencv-2.4.9/include/opencv/cv.h"
#include "/home/lineo/opencv-2.4.9/include/opencv/highgui.h"
#include "/home/lineo/opencv-2.4.9/include/opencv2/opencv.hpp"


#include <cuda/pyrdown.h>
#include <cuda/pyrlk.h>

////////////////////////////////
/// \brief system::system
///
System::System()
{


//    const int ncases = 12;
//    const size_t widths[ncases] = { 5, 10, 20, 50, 70, 90, 100,
//                                    200, 500, 700, 900, 1000 };
//    const size_t height = 10;

//    float *vals[ncases];
//    size_t pitches[ncases];

//    struct cudaDeviceProp p;
//    cudaGetDeviceProperties(&p, 0);
//    fprintf(stdout, "Texture alignment = %zd bytes\n",
//            p.textureAlignment);
//    cudaSetDevice(0);
//    cudaFree(0); // establish context

//    for(int i=0; i<ncases; i++) {
//        cudaMallocPitch((void **)&vals[i], &pitches[i],
//                        widths[i], height);
//        fprintf(stdout, "width = %zd <=> pitch = %zd \n",
//                widths[i], pitches[i]);
//    }


    // Opening the camera
    videoSource.open(0);
    if(!videoSource.isOpened())
    {
        // Can open the video source
    }

    // Create the buffer for the RGB in image
    //imageRGB.create(480,640,CV_8UC3);

}


/////////////////////////////////
/// \brief system::run
///
void System::run(void)
{

    Mat imageGrayL1 = cv::imread("/home/lineo/Bureau/Developpement/Cuda/Projet1/data/minicooper/frame10.pgm",0);
    Mat imageGrayL2 = cv::imread("/home/lineo/Bureau/Developpement/Cuda/Projet1/data/minicooper/frame11.pgm",0);

    PyrLK_gpu *ptPyrLK = new PyrLK_gpu();
    ptPyrLK->run_sparse(imageGrayL1.data,imageGrayL2.data,imageGrayL1.rows,imageGrayL1.cols);

}

////////////////////////////////
/// \brief system::grabImage
///
void System::grabImage(void)
{
    videoSource >> imageRGB;
    cvtColor(imageRGB, imageGray, CV_BGR2GRAY);
}
