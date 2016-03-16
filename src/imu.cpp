#include "include/imu.h"

imu::imu()
{

}

void imu::setAccelerometer()
{

}

void imu::setGyrometer()
{

}

void imu::setMagnetometer()
{

}

void imu::setSensor(float x,float y,float z,TYPE_SENSOR eSensorType)
{

        switch(eSensorType)
        {
        case ACCELEROMETER:
            ax = x;
            ay = y;
            az = z;
            setAccelerometer();
            break;
        case GYRO:
            gyrox = x;
            gyroy = y;
            gyroz = z;
            setGyrometer();
            break;
        case MAGNETOMETER:
            magx = x;
            magy = y;
            magz = z;
            setMagnetometer();
            break;

        }

}
