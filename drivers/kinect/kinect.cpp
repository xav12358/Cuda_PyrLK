#include "kinect.h"

//---------------------------------------------------------------------------
// Defines
//---------------------------------------------------------------------------
#define SAMPLE_XML_PATH "/home/xavier/kinect/OpenNI/Samples/Config/SamplesConfig.xml"

//---------------------------------------------------------------------------
// Macros
//---------------------------------------------------------------------------
#define CHECK_RC(rc, what) \
    if (rc != XN_STATUS_OK)\
{\
    printf("%s failed: %s\n", what, xnGetStatusString(rc));\
    }
//                            return rc;  \
//                             }

#include "iostream"
using namespace std;

kinect::kinect()
{

    XnStatus nRetVal = XN_STATUS_OK;
    nRetVal = context.InitFromXmlFile(SAMPLE_XML_PATH, scriptNode, &errors);
    nRetVal = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depth);
    CHECK_RC(nRetVal, "Find depth generator");

    XnFPSData xnFPS;
    nRetVal = xnFPSInit(&xnFPS, 180);
    CHECK_RC(nRetVal, "FPS Init");

    mvSize[0] = 640;
    mvSize[1] = 480;

    // définition du centre de l'image
    mvCenter = mvSize /2;

    // définition de la focal de la caméra
    mvFocal[0]  = 800;
    mvFocal[1]  = 800;

}

kinect::~kinect()
{
    depth.Release();
    scriptNode.Release();
    context.Release();
}


void kinect::GetFrame()
{
    static bool bFirstFrame = 1;
    XnStatus nRetVal = context.WaitOneUpdateAll(depth);

    if (nRetVal != XN_STATUS_OK)
    {
        printf("UpdateData failed: %s\n", xnGetStatusString(nRetVal));
        //        continue;
    }

    depth.GetMetaData(depthMD);

    if(bFirstFrame)
    {
        bFirstFrame = 0;
        LastImg     = new Mat(depthMD.YRes(),depthMD.XRes(),CV_16U);
    }

    Mat *img = new Mat(depthMD.YRes(),depthMD.XRes(),CV_16U);
    img->data = (uchar *)depthMD.Data();
    img->copyTo(*LastImg);


    imshow("Image kinect ",*LastImg);
    waitKey(20);

    delete img;
}


TooN::Vector<3> kinect::UnProject(int u,int v)
{
    TooN::Vector<3> V3DPos;

    V3DPos[2] = LastImg->at<u_int16_t>(v,u)/1000.0;

    ////////////////////////////////////////
    // u = f_x * X/Z + u0
    // v = f_y * Y/Z + v0
    if(V3DPos[2] !=0)
    {
        V3DPos[0] = (u - mvCenter[0])*V3DPos[2]/mvFocal[0];
        V3DPos[1] = (v - mvCenter[1])*V3DPos[2]/mvFocal[1];
    }

    return V3DPos;

}

void kinect::UnProjectAllImage()
{
    static bool bFirstUnProject = true;

    if(bFirstUnProject)
    {
        VTab3DImage = (TooN::Vector<3> *)malloc(mvSize[0]*mvSize[1]*sizeof(TooN::Vector<3>));
        bFirstUnProject = false;
    }

    int iStep = mvSize[1];
    for(int u =0; u<mvSize[0] ; u++)
        for(int v =0; v<mvSize[1]; v++)
        {
            VTab3DImage[u + v*iStep] = UnProject(u,v);
        }
}
