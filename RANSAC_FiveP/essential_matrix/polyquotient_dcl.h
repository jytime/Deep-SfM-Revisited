#ifndef _polyquotient_cc_dcl_
#define _polyquotient_cc_dcl_

#include "common.h"

__host__ __device__ void polyquotient (
   double *a, int sa, 
   double *b, double *t, int sb, 
   double *q, int &sq,
   BMatrix B, int &current_size
   );

#endif
