#include <include/system.h>
#include "/home/lineo/opencv-2.4.9/include/opencv/cv.h"
#include "/home/lineo/opencv-2.4.9/include/opencv/highgui.h"
#include "/home/lineo/opencv-2.4.9/include/opencv2/opencv.hpp"

#include <QDebug>

#include <cuda/pyrdown.h>
#include <cuda/pyrlk.h>
#include <cuda/fast.h>

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
    Mat imageGrayL2 = cv::imread("/home/lineo/Bureau/Developpement/Cuda/Projet1/data/minicooper/frame10.pgm",0);

    //PyrLK_gpu *ptPyrLK = new PyrLK_gpu();
    //ptPyrLK->run_sparse(imageGrayL1.data,imageGrayL2.data,imageGrayL1.rows,imageGrayL1.cols);

    int MaxKeypoints = 15000;
    Fast_gpu *Fastgpu = new Fast_gpu(imageGrayL1.cols,imageGrayL1.rows,MaxKeypoints);
    int nbkeypoints = Fastgpu->run_calcKeypoints(imageGrayL1.data,45);
    qDebug() << "nbkeypoints " << nbkeypoints;
    short2* kpLoc = new short2[nbkeypoints];

    checkCudaErrors(cudaMemcpy(kpLoc , Fastgpu->kpLoc, nbkeypoints* sizeof(short2), cudaMemcpyDeviceToHost));

    for(int j=0;j<nbkeypoints;j++)
        cv::circle(imageGrayL1,cv::Point( kpLoc[j].x, kpLoc[j].y ),3,cv::Scalar(0,0,255),2);


    cv::imshow("Image with keypoints",imageGrayL1);



    int nbkeypoints_nonmaxSuppression = Fastgpu->run_nonmaxSuppression(nbkeypoints);
    qDebug() << "nbkeypoints_nonmaxSuppression " << nbkeypoints_nonmaxSuppression;
    short2* kpLocFinal = new short2[nbkeypoints_nonmaxSuppression];
    checkCudaErrors(cudaMemcpy(kpLocFinal , Fastgpu->kpLocFinal, nbkeypoints_nonmaxSuppression* sizeof(short2), cudaMemcpyDeviceToHost));

    for(int j=0;j<nbkeypoints_nonmaxSuppression;j++)
        cv::circle(imageGrayL2,cv::Point( kpLocFinal[j].x, kpLocFinal[j].y ),3,cv::Scalar(0,0,255),2);
    cv::imshow("Image with keypoints nonmaxSuppression ",imageGrayL2);
    cv::waitKey(-1);


}

////////////////////////////////
/// \brief system::grabImage
///
void System::grabImage(void)
{
    videoSource >> imageRGB;
    cvtColor(imageRGB, imageGray, CV_BGR2GRAY);
}
