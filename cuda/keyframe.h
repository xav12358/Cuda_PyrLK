#ifndef PYRLK_H
#define PYRLK_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cuda_runtime_api.h>    // includes cuda.h and cuda_runtime_api.h
// Each keyframe is made of LEVELS pyramid levels, stored in struct Level.
// This contains image data and corner points.
class Level
{
    Level();



    u_int8_t ptData;                // The pyramid level pixels
    std::vector<float2> vCorners;     // All FAST corners on this level
    //std::vector<int> vCornerRowLUT;          // Row-index into the FAST corners, speeds up access
    std::vector<float2> vMaxCorners;  // The maximal FAST corners
    Level& operator=(const Level &rhs);

    //std::vector<Candidate> vCandidates;   // Potential locations of new map points

    bool bImplaneCornersCached;           // Also keep image-plane (z=1) positions of FAST corners to speed up epipolar search
    //std::vector<TooN::Vector<2> > vImplaneCorners; // Corner points un-projected into z=1-plane coordinates
};


class KeyFrame
{

public:


    KeyFrame();
    ~KeyFrame();

};

#endif // PYRLK_H
