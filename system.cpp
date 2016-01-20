#include "system.h"

#include "/home/lineo/opencv-2.4.9/include/opencv/cv.h"
#include "/home/lineo/opencv-2.4.9/include/opencv/highgui.h"
#include "/home/lineo/opencv-2.4.9/include/opencv2/opencv.hpp"


#include <pyrdown.h>
#include <pyrlk.h>

////////////////////////////////
/// \brief system::system
///
System::System()
{

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
