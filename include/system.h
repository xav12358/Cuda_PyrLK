#ifndef SYSTEM_H
#define SYSTEM_H

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;

class System
{


    VideoCapture videoSource;    //< video source
    Mat imageRGB;               //< RGB Image from the video source
    Mat imageGray;              //< Gray Image from the video source


public:
    System();
    void grabImage(void);
    void run(void);
};

#endif // SYSTEM_H
