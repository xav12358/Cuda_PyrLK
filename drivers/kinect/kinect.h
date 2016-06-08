#ifndef KINECT_H
#define KINECT_H


#include <XnOpenNI.h>
#include <XnLog.h>
#include <XnCppWrapper.h>
#include <XnFPSCalculator.h>
using namespace xn;


#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;

#include <TooN/TooN.h>
using namespace TooN;

class kinect
{
public:

    Context context;
    ScriptNode scriptNode;
    EnumerationErrors errors;
    DepthGenerator depth;
    DepthMetaData depthMD;

    TooN::Vector<2>mvCenter;     // Pixel projection center
    TooN::Vector<2>mvFocal;      // Pixel focal length
    TooN::Vector<2>mvSize;       // Image size

    TooN::Vector<3> * VTab3DImage;

    Mat *LastImg;

    kinect();
    ~kinect();
    void GetFrame();
    TooN::Vector<3> UnProject(int u, int v);
    void UnProjectAllImage();
};

#endif // KINECT_H
