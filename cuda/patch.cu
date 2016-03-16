#include "patch.h"

#define SIZE_X 8
#define SIZE_Y 8

#define SIZE_X_WITH_BORDER SIZE_X+2
#define SIZE_Y_WITH_BORDER SIZE_Y+2

patch::patch()
{
    ptPatchWithBorder = (u_int8_t *)malloc(SIZE_X_WITH_BORDER*SIZE_Y_WITH_BORDER);
}

