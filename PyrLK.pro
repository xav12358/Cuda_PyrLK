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
    src/system.cpp \
    src/tracker.cpp \
    src/point.cpp \
    src/pthreadcpp.cpp \
    src/imu.cpp



HEADERS += \
    include/system.h \
    cuda/global_var.h \
    cuda/keyframe.h \
    cuda/fast.h \
    cuda/patchtracker.h \
    cuda/pyrdown.h \
    cuda/pyrlk.h \
    cuda/patch.h\
    include/tracker.h \
    include/point.h \
    include/imu.h \
    include/Eigen/src/Cholesky/LDLT.h \
    include/Eigen/src/Cholesky/LLT.h \
    include/Eigen/src/Cholesky/LLT_MKL.h \
    include/Eigen/src/CholmodSupport/CholmodSupport.h \
    include/Eigen/src/Core/arch/AltiVec/Complex.h \
    include/Eigen/src/Core/arch/AltiVec/PacketMath.h \
    include/Eigen/src/Core/arch/Default/Settings.h \
    include/Eigen/src/Core/arch/NEON/Complex.h \
    include/Eigen/src/Core/arch/NEON/PacketMath.h \
    include/Eigen/src/Core/arch/SSE/Complex.h \
    include/Eigen/src/Core/arch/SSE/MathFunctions.h \
    include/Eigen/src/Core/arch/SSE/PacketMath.h \
    include/Eigen/src/Core/products/CoeffBasedProduct.h \
    include/Eigen/src/Core/products/GeneralBlockPanelKernel.h \
    include/Eigen/src/Core/products/GeneralMatrixMatrix.h \
    include/Eigen/src/Core/products/GeneralMatrixMatrix_MKL.h \
    include/Eigen/src/Core/products/GeneralMatrixMatrixTriangular.h \
    include/Eigen/src/Core/products/GeneralMatrixMatrixTriangular_MKL.h \
    include/Eigen/src/Core/products/GeneralMatrixVector.h \
    include/Eigen/src/Core/products/GeneralMatrixVector_MKL.h \
    include/Eigen/src/Core/products/Parallelizer.h \
    include/Eigen/src/Core/products/SelfadjointMatrixMatrix.h \
    include/Eigen/src/Core/products/SelfadjointMatrixMatrix_MKL.h \
    include/Eigen/src/Core/products/SelfadjointMatrixVector.h \
    include/Eigen/src/Core/products/SelfadjointMatrixVector_MKL.h \
    include/Eigen/src/Core/products/SelfadjointProduct.h \
    include/Eigen/src/Core/products/SelfadjointRank2Update.h \
    include/Eigen/src/Core/products/TriangularMatrixMatrix.h \
    include/Eigen/src/Core/products/TriangularMatrixMatrix_MKL.h \
    include/Eigen/src/Core/products/TriangularMatrixVector.h \
    include/Eigen/src/Core/products/TriangularMatrixVector_MKL.h \
    include/Eigen/src/Core/products/TriangularSolverMatrix.h \
    include/Eigen/src/Core/products/TriangularSolverMatrix_MKL.h \
    include/Eigen/src/Core/products/TriangularSolverVector.h \
    include/Eigen/src/Core/util/BlasUtil.h \
    include/Eigen/src/Core/util/Constants.h \
    include/Eigen/src/Core/util/DisableStupidWarnings.h \
    include/Eigen/src/Core/util/ForwardDeclarations.h \
    include/Eigen/src/Core/util/Macros.h \
    include/Eigen/src/Core/util/Memory.h \
    include/Eigen/src/Core/util/Meta.h \
    include/Eigen/src/Core/util/MKL_support.h \
    include/Eigen/src/Core/util/NonMPL2.h \
    include/Eigen/src/Core/util/ReenableStupidWarnings.h \
    include/Eigen/src/Core/util/StaticAssert.h \
    include/Eigen/src/Core/util/XprHelper.h \
    include/Eigen/src/Core/Array.h \
    include/Eigen/src/Core/ArrayBase.h \
    include/Eigen/src/Core/ArrayWrapper.h \
    include/Eigen/src/Core/Assign.h \
    include/Eigen/src/Core/Assign_MKL.h \
    include/Eigen/src/Core/BandMatrix.h \
    include/Eigen/src/Core/Block.h \
    include/Eigen/src/Core/BooleanRedux.h \
    include/Eigen/src/Core/CommaInitializer.h \
    include/Eigen/src/Core/CoreIterators.h \
    include/Eigen/src/Core/CwiseBinaryOp.h \
    include/Eigen/src/Core/CwiseNullaryOp.h \
    include/Eigen/src/Core/CwiseUnaryOp.h \
    include/Eigen/src/Core/CwiseUnaryView.h \
    include/Eigen/src/Core/DenseBase.h \
    include/Eigen/src/Core/DenseCoeffsBase.h \
    include/Eigen/src/Core/DenseStorage.h \
    include/Eigen/src/Core/Diagonal.h \
    include/Eigen/src/Core/DiagonalMatrix.h \
    include/Eigen/src/Core/DiagonalProduct.h \
    include/Eigen/src/Core/Dot.h \
    include/Eigen/src/Core/EigenBase.h \
    include/Eigen/src/Core/Flagged.h \
    include/Eigen/src/Core/ForceAlignedAccess.h \
    include/Eigen/src/Core/Functors.h \
    include/Eigen/src/Core/Fuzzy.h \
    include/Eigen/src/Core/GeneralProduct.h \
    include/Eigen/src/Core/GenericPacketMath.h \
    include/Eigen/src/Core/GlobalFunctions.h \
    include/Eigen/src/Core/IO.h \
    include/Eigen/src/Core/Map.h \
    include/Eigen/src/Core/MapBase.h \
    include/Eigen/src/Core/MathFunctions.h \
    include/Eigen/src/Core/Matrix.h \
    include/Eigen/src/Core/MatrixBase.h \
    include/Eigen/src/Core/NestByValue.h \
    include/Eigen/src/Core/NoAlias.h \
    include/Eigen/src/Core/NumTraits.h \
    include/Eigen/src/Core/PermutationMatrix.h \
    include/Eigen/src/Core/PlainObjectBase.h \
    include/Eigen/src/Core/ProductBase.h \
    include/Eigen/src/Core/Random.h \
    include/Eigen/src/Core/Redux.h \
    include/Eigen/src/Core/Ref.h \
    include/Eigen/src/Core/Replicate.h \
    include/Eigen/src/Core/ReturnByValue.h \
    include/Eigen/src/Core/Reverse.h \
    include/Eigen/src/Core/Select.h \
    include/Eigen/src/Core/SelfAdjointView.h \
    include/Eigen/src/Core/SelfCwiseBinaryOp.h \
    include/Eigen/src/Core/SolveTriangular.h \
    include/Eigen/src/Core/StableNorm.h \
    include/Eigen/src/Core/Stride.h \
    include/Eigen/src/Core/Swap.h \
    include/Eigen/src/Core/Transpose.h \
    include/Eigen/src/Core/Transpositions.h \
    include/Eigen/src/Core/TriangularMatrix.h \
    include/Eigen/src/Core/VectorBlock.h \
    include/Eigen/src/Core/VectorwiseOp.h \
    include/Eigen/src/Core/Visitor.h \
    include/Eigen/src/Eigen2Support/Geometry/AlignedBox.h \
    include/Eigen/src/Eigen2Support/Geometry/All.h \
    include/Eigen/src/Eigen2Support/Geometry/AngleAxis.h \
    include/Eigen/src/Eigen2Support/Geometry/Hyperplane.h \
    include/Eigen/src/Eigen2Support/Geometry/ParametrizedLine.h \
    include/Eigen/src/Eigen2Support/Geometry/Quaternion.h \
    include/Eigen/src/Eigen2Support/Geometry/Rotation2D.h \
    include/Eigen/src/Eigen2Support/Geometry/RotationBase.h \
    include/Eigen/src/Eigen2Support/Geometry/Scaling.h \
    include/Eigen/src/Eigen2Support/Geometry/Transform.h \
    include/Eigen/src/Eigen2Support/Geometry/Translation.h \
    include/Eigen/src/Eigen2Support/Block.h \
    include/Eigen/src/Eigen2Support/Cwise.h \
    include/Eigen/src/Eigen2Support/CwiseOperators.h \
    include/Eigen/src/Eigen2Support/Lazy.h \
    include/Eigen/src/Eigen2Support/LeastSquares.h \
    include/Eigen/src/Eigen2Support/LU.h \
    include/Eigen/src/Eigen2Support/Macros.h \
    include/Eigen/src/Eigen2Support/MathFunctions.h \
    include/Eigen/src/Eigen2Support/Memory.h \
    include/Eigen/src/Eigen2Support/Meta.h \
    include/Eigen/src/Eigen2Support/Minor.h \
    include/Eigen/src/Eigen2Support/QR.h \
    include/Eigen/src/Eigen2Support/SVD.h \
    include/Eigen/src/Eigen2Support/TriangularSolver.h \
    include/Eigen/src/Eigen2Support/VectorBlock.h \
    include/Eigen/src/Eigenvalues/ComplexEigenSolver.h \
    include/Eigen/src/Eigenvalues/ComplexSchur.h \
    include/Eigen/src/Eigenvalues/ComplexSchur_MKL.h \
    include/Eigen/src/Eigenvalues/EigenSolver.h \
    include/Eigen/src/Eigenvalues/GeneralizedEigenSolver.h \
    include/Eigen/src/Eigenvalues/GeneralizedSelfAdjointEigenSolver.h \
    include/Eigen/src/Eigenvalues/HessenbergDecomposition.h \
    include/Eigen/src/Eigenvalues/MatrixBaseEigenvalues.h \
    include/Eigen/src/Eigenvalues/RealQZ.h \
    include/Eigen/src/Eigenvalues/RealSchur.h \
    include/Eigen/src/Eigenvalues/RealSchur_MKL.h \
    include/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h \
    include/Eigen/src/Eigenvalues/SelfAdjointEigenSolver_MKL.h \
    include/Eigen/src/Eigenvalues/Tridiagonalization.h \
    include/Eigen/src/Geometry/arch/Geometry_SSE.h \
    include/Eigen/src/Geometry/AlignedBox.h \
    include/Eigen/src/Geometry/AngleAxis.h \
    include/Eigen/src/Geometry/EulerAngles.h \
    include/Eigen/src/Geometry/Homogeneous.h \
    include/Eigen/src/Geometry/Hyperplane.h \
    include/Eigen/src/Geometry/OrthoMethods.h \
    include/Eigen/src/Geometry/ParametrizedLine.h \
    include/Eigen/src/Geometry/Quaternion.h \
    include/Eigen/src/Geometry/Rotation2D.h \
    include/Eigen/src/Geometry/RotationBase.h \
    include/Eigen/src/Geometry/Scaling.h \
    include/Eigen/src/Geometry/Transform.h \
    include/Eigen/src/Geometry/Translation.h \
    include/Eigen/src/Geometry/Umeyama.h \
    include/Eigen/src/Householder/BlockHouseholder.h \
    include/Eigen/src/Householder/Householder.h \
    include/Eigen/src/Householder/HouseholderSequence.h \
    include/Eigen/src/IterativeLinearSolvers/BasicPreconditioners.h \
    include/Eigen/src/IterativeLinearSolvers/BiCGSTAB.h \
    include/Eigen/src/IterativeLinearSolvers/ConjugateGradient.h \
    include/Eigen/src/IterativeLinearSolvers/IncompleteLUT.h \
    include/Eigen/src/IterativeLinearSolvers/IterativeSolverBase.h \
    include/Eigen/src/Jacobi/Jacobi.h \
    include/Eigen/src/LU/arch/Inverse_SSE.h \
    include/Eigen/src/LU/Determinant.h \
    include/Eigen/src/LU/FullPivLU.h \
    include/Eigen/src/LU/Inverse.h \
    include/Eigen/src/LU/PartialPivLU.h \
    include/Eigen/src/LU/PartialPivLU_MKL.h \
    include/Eigen/src/MetisSupport/MetisSupport.h \
    include/Eigen/src/misc/blas.h \
    include/Eigen/src/misc/Image.h \
    include/Eigen/src/misc/Kernel.h \
    include/Eigen/src/misc/Solve.h \
    include/Eigen/src/misc/SparseSolve.h \
    include/Eigen/src/OrderingMethods/Amd.h \
    include/Eigen/src/OrderingMethods/Eigen_Colamd.h \
    include/Eigen/src/OrderingMethods/Ordering.h \
    include/Eigen/src/PardisoSupport/PardisoSupport.h \
    include/Eigen/src/PaStiXSupport/PaStiXSupport.h \
    include/Eigen/src/plugins/ArrayCwiseBinaryOps.h \
    include/Eigen/src/plugins/ArrayCwiseUnaryOps.h \
    include/Eigen/src/plugins/BlockMethods.h \
    include/Eigen/src/plugins/CommonCwiseBinaryOps.h \
    include/Eigen/src/plugins/CommonCwiseUnaryOps.h \
    include/Eigen/src/plugins/MatrixCwiseBinaryOps.h \
    include/Eigen/src/plugins/MatrixCwiseUnaryOps.h \
    include/Eigen/src/QR/ColPivHouseholderQR.h \
    include/Eigen/src/QR/ColPivHouseholderQR_MKL.h \
    include/Eigen/src/QR/FullPivHouseholderQR.h \
    include/Eigen/src/QR/HouseholderQR.h \
    include/Eigen/src/QR/HouseholderQR_MKL.h \
    include/Eigen/src/SparseCholesky/SimplicialCholesky.h \
    include/Eigen/src/SparseCholesky/SimplicialCholesky_impl.h \
    include/Eigen/src/SparseCore/AmbiVector.h \
    include/Eigen/src/SparseCore/CompressedStorage.h \
    include/Eigen/src/SparseCore/ConservativeSparseSparseProduct.h \
    include/Eigen/src/SparseCore/MappedSparseMatrix.h \
    include/Eigen/src/SparseCore/SparseBlock.h \
    include/Eigen/src/SparseCore/SparseColEtree.h \
    include/Eigen/src/SparseCore/SparseCwiseBinaryOp.h \
    include/Eigen/src/SparseCore/SparseCwiseUnaryOp.h \
    include/Eigen/src/SparseCore/SparseDenseProduct.h \
    include/Eigen/src/SparseCore/SparseDiagonalProduct.h \
    include/Eigen/src/SparseCore/SparseDot.h \
    include/Eigen/src/SparseCore/SparseFuzzy.h \
    include/Eigen/src/SparseCore/SparseMatrix.h \
    include/Eigen/src/SparseCore/SparseMatrixBase.h \
    include/Eigen/src/SparseCore/SparsePermutation.h \
    include/Eigen/src/SparseCore/SparseProduct.h \
    include/Eigen/src/SparseCore/SparseRedux.h \
    include/Eigen/src/SparseCore/SparseSelfAdjointView.h \
    include/Eigen/src/SparseCore/SparseSparseProductWithPruning.h \
    include/Eigen/src/SparseCore/SparseTranspose.h \
    include/Eigen/src/SparseCore/SparseTriangularView.h \
    include/Eigen/src/SparseCore/SparseUtil.h \
    include/Eigen/src/SparseCore/SparseVector.h \
    include/Eigen/src/SparseCore/SparseView.h \
    include/Eigen/src/SparseCore/TriangularSolver.h \
    include/Eigen/src/SparseLU/SparseLU.h \
    include/Eigen/src/SparseLU/SparseLU_column_bmod.h \
    include/Eigen/src/SparseLU/SparseLU_column_dfs.h \
    include/Eigen/src/SparseLU/SparseLU_copy_to_ucol.h \
    include/Eigen/src/SparseLU/SparseLU_gemm_kernel.h \
    include/Eigen/src/SparseLU/SparseLU_heap_relax_snode.h \
    include/Eigen/src/SparseLU/SparseLU_kernel_bmod.h \
    include/Eigen/src/SparseLU/SparseLU_Memory.h \
    include/Eigen/src/SparseLU/SparseLU_panel_bmod.h \
    include/Eigen/src/SparseLU/SparseLU_panel_dfs.h \
    include/Eigen/src/SparseLU/SparseLU_pivotL.h \
    include/Eigen/src/SparseLU/SparseLU_pruneL.h \
    include/Eigen/src/SparseLU/SparseLU_relax_snode.h \
    include/Eigen/src/SparseLU/SparseLU_Structs.h \
    include/Eigen/src/SparseLU/SparseLU_SupernodalMatrix.h \
    include/Eigen/src/SparseLU/SparseLU_Utils.h \
    include/Eigen/src/SparseLU/SparseLUImpl.h \
    include/Eigen/src/SparseQR/SparseQR.h \
    include/Eigen/src/SPQRSupport/SuiteSparseQRSupport.h \
    include/Eigen/src/StlSupport/details.h \
    include/Eigen/src/StlSupport/StdDeque.h \
    include/Eigen/src/StlSupport/StdList.h \
    include/Eigen/src/StlSupport/StdVector.h \
    include/Eigen/src/SuperLUSupport/SuperLUSupport.h \
    include/Eigen/src/SVD/JacobiSVD.h \
    include/Eigen/src/SVD/JacobiSVD_MKL.h \
    include/Eigen/src/SVD/UpperBidiagonalization.h \
    include/Eigen/src/UmfPackSupport/UmfPackSupport.h \
    patch.h \
    include/pthreadcpp.h



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



# CUDA settings <-- may change depending on your system
CUDA_SOURCES += ./cuda/keyframe.cu
CUDA_SOURCES += ./cuda/fast.cu
CUDA_SOURCES += ./cuda/pyrdown.cu
CUDA_SOURCES += ./cuda/pyrlk.cu
CUDA_SOURCES += ./cuda/patchtracker.cu
CUDA_SOURCES += ./cuda/patch.cu


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

DISTFILES += \
    patchtracker.cu



