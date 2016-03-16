#ifndef IMU_H
#define IMU_H


#include <include/Eigen/Dense>


//using namespace Eigen;
class imu
{

    typedef enum{
        ACCELEROMETER,
        GYRO,
        MAGNETOMETER
    }TYPE_SENSOR;

    Eigen::Quaterniond qSE;

    float ax,ay,az;
    float gyrox,gyroy,gyroz;
    float magx,magy,magz;
public:
    imu();
    ~imu();

    void setAccelerometer();
    void setGyrometer();
    void setMagnetometer();
    void setSensor(float x,float y,float z,TYPE_SENSOR eSensorType);

};

#endif // IMU_H
