#include <include/system.h>
#include "/home/lineo/opencv-2.4.9/include/opencv/cv.h"
#include "/home/lineo/opencv-2.4.9/include/opencv/highgui.h"
#include "/home/lineo/opencv-2.4.9/include/opencv2/opencv.hpp"

#include <QDebug>

#include <cuda/fast.h>
#include <cuda/keyframe.h>
#include <cuda/patchtracker.h>
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
    //videoSource.open(0);
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
    ///////////////////////////////////////////////////////
    //////////////// Test of the pyrdown algorithm/////////
    ///
    Mat imageGrayL1 = cv::imread("/home/lineo/Bureau/Developpement/Cuda/Projet1/data/minicooper/frame11.pgm",0);
    PyrDown_gpu *ptPyrDown = new PyrDown_gpu(imageGrayL1.rows,imageGrayL1.cols);
    ptPyrDown->run(imageGrayL1.rows,imageGrayL1.cols,imageGrayL1.data);
    
    
    
    ///////////////////////////////////////////////////////
    //////////////// Test of the fast algorithm////////////
    ///
    //    Mat imageGrayL1 = cv::imread("/home/lineo/Bureau/Developpement/Cuda/Projet1/data/minicooper/frame11.pgm",0);
    //    Mat imageGrayL2 = cv::imread("/home/lineo/Bureau/Developpement/Cuda/Projet1/data/minicooper/frame10.pgm",0);
    
    
    //    int MaxKeypoints = 25000;
    //    Fast_gpu *Fastgpu = new Fast_gpu(imageGrayL1.cols,imageGrayL1.rows,MaxKeypoints);
    //    int nbkeypoints = Fastgpu->run_calcKeypoints(imageGrayL1.data,25);
    //    qDebug() << "nbkeypoints " << nbkeypoints;
    //    short2* kpLoc = new short2[nbkeypoints];
    
    //    checkCudaErrors(cudaMemcpy(kpLoc , Fastgpu->kpLoc, nbkeypoints* sizeof(short2), cudaMemcpyDeviceToHost));
    
    //    for(int j=0;j<nbkeypoints;j++)
    //        cv::circle(imageGrayL1,cv::Point( kpLoc[j].x, kpLoc[j].y ),3,cv::Scalar(0,0,255),2);
    //    cv::imshow("Image with keypoints",imageGrayL1);
    
    
    //    int nbkeypoints_nonmaxSuppression = Fastgpu->run_nonmaxSuppression(nbkeypoints);
    //    qDebug() << "nbkeypoints_nonmaxSuppression " << nbkeypoints_nonmaxSuppression;
    //    short2* kpLocFinal = new short2[nbkeypoints_nonmaxSuppression];
    //    checkCudaErrors(cudaMemcpy(kpLocFinal , Fastgpu->kpLocFinal, nbkeypoints_nonmaxSuppression* sizeof(short2), cudaMemcpyDeviceToHost));
    
    //    for(int j=0;j<nbkeypoints_nonmaxSuppression;j++)
    //        cv::circle(imageGrayL2,cv::Point( kpLocFinal[j].x, kpLocFinal[j].y ),3,cv::Scalar(0,0,255),2);
    //    cv::imshow("Image with keypoints nonmaxSuppression ",imageGrayL2);
    //    cv::waitKey(-1);
    
    //    double time = cv::getTickCount();
    //    imwrite("image.png",imageGrayL2);
    //    std::cout << "time imwrite: " << (cv::getTickCount()-time)/cv::getTickFrequency() * 1000.0 << std::endl;
    
    
    ///////////////////////////////////////////////////////
    //////////////// Test of the pyrdown       ////////////
    ///
    //Mat imageGrayL1 = cv::imread("/home/lineo/Bureau/Developpement/Cuda/Projet1/data/minicooper/frame11.pgm",0);
    /*KeyFrame *ptKeyFrame = new KeyFrame(imageGrayL1.rows,imageGrayL1.cols,ToCalculate);
    qDebug() << "i : " << imageGrayL1.cols << " " << imageGrayL1.rows;
    for(int i=0;i<4;i++)
    {
        qDebug() << "i : " << ptKeyFrame->Levels[i].icol << " " << ptKeyFrame->Levels[i].irow;
    }
    
    qDebug() << "MakePyramid ";
    ptKeyFrame->MakePyramid(imageGrayL1.data);
    qDebug() << "MakePyramid_end ";
    
    
    for(int i=0;i<4;i++)
    {
        cv::Mat *ImageL = new cv::Mat(ptKeyFrame->Levels[i].irow,ptKeyFrame->Levels[i].icol,CV_8U,ptKeyFrame->Levels[i].u8_ptData);
        cv::Mat *ImageLMax = new cv::Mat(ptKeyFrame->Levels[i].irow,ptKeyFrame->Levels[i].icol,CV_8U);
        
        ImageL->copyTo(*ImageLMax);
        qDebug() << "NbKeyframe " << ptKeyFrame->Levels[i].iNbKeypoints;
        //        for(int j=0;j<ptKeyFrame->Levels[i].iNbKeypoints && j<1000;j++)
        //        {
        //            cv::circle(*ImageL,cv::Point( ptKeyFrame->Levels[i].kpLoc[j].x, ptKeyFrame->Levels[i].kpLoc[j].y ),3,cv::Scalar(0,0,255),2);
        //        }
        
        qDebug() << "NbKeyframeMax  " << ptKeyFrame->Levels[i].iNbKeypointsMax;
        for(int j=0;j<ptKeyFrame->Levels[i].iNbKeypointsMax && j<10000;j++)
        {
            cv::circle(*ImageLMax,cv::Point( ptKeyFrame->Levels[i].kpLocMax[j].x, ptKeyFrame->Levels[i].kpLocMax[j].y ),2,cv::Scalar(0,0,255),2);
        }
        imshow("Level",*ImageL);
        imshow("LevelMax",*ImageLMax);
        cv::waitKey(-1);
    }*/
    


    // uint8_t *ptData = new uint8_t[imageGrayL1.rows*imageGrayL1.cols];
    //ptData = (uint8_t *)malloc(imageGrayL1.rows*imageGrayL1.cols*sizeof(u_int8_t));
    
    
    //checkCudaErrors(cudaMalloc((void **)&ptKeyFrame->ptTmpBloc,  640*480 * sizeof(u_int8_t)));
    //    for(int i=1;i<4;i++)
    //    {
    //        cv::Mat Image2(ptKeyFrame->Levels[i].irow,ptKeyFrame->Levels[i].icol,CV_8U);
    
    //        std::cout << "ilevel " << i << " irow "<< ptKeyFrame->Levels[i].irow<<" icol" <<ptKeyFrame->Levels[i].icol <<std::endl;
    //        checkCudaErrors( cudaMemcpy(Image2.data, ptKeyFrame->Levels[i].ptData , ptKeyFrame->Levels[i].irow*ptKeyFrame->Levels[i].icol*sizeof(u_int8_t),  cudaMemcpyDeviceToHost) );
    
    //        //cv::Mat Image2(ptKeyFrame->Levels[i].irow,ptKeyFrame->Levels[i].icol,CV_8U,ptData);
    //        //imshow("Resultat Level",Image2);
    //        //cv::waitKey(-1);
    //    }



    int halfpatch_size  = 5;
    int patch_size = 2*halfpatch_size+1;

    float minx = 0,maxx = 0,miny = 0,maxy = 0;
    for (int y=0; y<patch_size; ++y)
    {
        for (int x=0; x<patch_size; ++x )
        {
            float indexx = x-halfpatch_size;
            float  indexy = y-halfpatch_size;

            float valx = indexx * cos(M_PI/4) +indexy * sin(M_PI/4);
            float valy = -indexx * sin(M_PI/4) +indexy * cos(M_PI/4) ;

            if(valx<minx)
            {
                minx = valx;
            }

            if(valy<miny)
            {
                miny = valy;
            }

            if(valx>maxx)
            {
                maxx = valx;
            }

            if(valy>maxy)
            {
                maxy = valy;
            }
            std::cout << "valx " << valx <<" valy" << valy << std::endl;
            //std::cout << "(" << indexx << " , " << indexy << ")";
        }

        //std::cout << std::endl;
    }
    printf("min x :%f y:%f, max x:%f y :%f",minx,miny,maxx,maxy);


  /*  PatchTracker * ptTracker = new PatchTracker();
    qDebug() << "ptTracker  0";
    for(int i=0;i<120;i++)
    {
        qDebug() << "ptTracker  in " <<ptTracker->indiceFeatures ;
        ptTracker->addPatchToWarp(imageGrayL1.data,imageGrayL1.rows,imageGrayL1.cols,250,25+i);
    }


    qDebug() << "ptTracker  2 " <<ptTracker->indiceFeatures ;
    Mat ImagePatch(PATCH_SIZE_MAX*120,PATCH_SIZE_MAX,CV_8U,ptTracker->ptlistPatchsMaxHost);
    qDebug() << "ptTracker  2.1";
    imshow("ImagePatch",ImagePatch);

    cv::waitKey(-1);*/

}



////////////////////////////////
/// \brief system::grabImage
///
void System::grabImage(void)
{
    videoSource >> imageRGB;
    cvtColor(imageRGB, imageGray, CV_BGR2GRAY);
}
