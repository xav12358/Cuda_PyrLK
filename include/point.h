#ifndef POINT_H
#define POINT_H


#include <include/Eigen/Dense>

class point
{


    Eigen::VectorXd v3Center_NC;             // Unit vector in Source-KF coords pointing at the patch center
    Eigen::VectorXd v3OneDownFromCenter_NC;  // Unit vector in Source-KF coords pointing towards one pixel down of the patch center
    Eigen::VectorXd v3OneRightFromCenter_NC; // Unit vector in Source-KF coords pointing towards one pixel right of the patch center
    Eigen::VectorXd v3Normal_NC;             // Unit vector in Source-KF coords indicating patch normal

    Eigen::VectorXd v3PixelDown_W;           // 3-Vector in World coords corresponding to a one-pixel move down the source image
    Eigen::VectorXd v3PixelRight_W;          // 3-Vector in World coords corresponding to a one-pixel move right the source image

public:
    point();
};

#endif // POINT_H
