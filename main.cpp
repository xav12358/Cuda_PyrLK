#include <QtCore/QCoreApplication>

#include <system.h>


int main(int argc, char *argv[])
{

    System *CurrentSys = new System();
    CurrentSys->run();
}
