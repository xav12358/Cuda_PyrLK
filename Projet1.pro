#-------------------------------------------------
#
# Project created by QtCreator 2013-04-17T16:30:33
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = QtCuda
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += src/main.cpp \
    src/system.cpp

HEADERS += \
    system.h

#######################################################################################

CONFIG += link_pkgconfig
PKGCONFIG += opencv

#LIBS +=-lcv -lhighgui -lstdc++ -lcxcore -lcvaux


INCLUDEPATH += /usr/include/opencv
LIBS += -L/usr/local/lib
LIBS += -L/usr/lib/x86_64-linux-gnu
LIBS += -lm
LIBS += -lopencv_core
LIBS += -lopencv_imgproc
LIBS += -lopencv_highgui
LIBS += -lopencv_objdetect
LIBS += -lopencv_calib3d





#######################################################################################
# This makes the .cu files appear in your project
OTHER_FILES +=  ./cuda/pyrdown.h
OTHER_FILES +=  ./cuda/pyrlk.h
OTHER_FILES +=  ./cuda/keyframe.h

# CUDA settings <-- may change depending on your system
CUDA_SOURCES += ./cuda/pyrdown.cu
CUDA_SOURCES += ./cuda/pyrlk.cu
CUDA_SOURCES += ./cuda/keyframe.cu
CUDA_SDK = /usr/lib/nvidia-cuda-toolkit             #/usr/include/   # Path to cuda SDK install
CUDA_DIR = /usr/lib/nvidia-cuda-toolkit             # Path to cuda toolkit install

# DO NOT EDIT BEYOND THIS UNLESS YOU KNOW WHAT YOU ARE DOING....

SYSTEM_NAME = unix         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64           # '32' or '64', depending on your system
CUDA_ARCH = compute_11#sm_21           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = #--use_fast_math


# include paths
INCLUDEPATH += $$CUDA_DIR/include

# library directories
#QMAKE_LIBDIR += /usr/lib/i386-linux-gnu#/usr/lib/nvidia-cuda-toolkit/lib #/usr/lib/i386-linux-gnu #$CUDA_DIR/lib/
QMAKE_LIBDIR += /usr/lib/x86_64-linux-gnu#/usr/lib/nvidia-cuda-toolkit/lib #/usr/lib/i386-linux-gnu #$CUDA_DIR/lib/


CUDA_OBJECTS_DIR = ./


# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += -L /usr/lib/x86_64-linux-gnu -lcuda -lcudart

# Configuration of the Cuda compiler
#CONFIG(debug, debug|release) {
#    # Debug mode
#    cuda_d.input = CUDA_SOURCES
#    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
#    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
#    cuda_d.dependency_type = TYPE_C
#    QMAKE_EXTRA_COMPILERS += cuda_d
#}
#else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
#}

DISTFILES +=



