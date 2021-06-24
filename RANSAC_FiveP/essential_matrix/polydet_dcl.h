#ifndef _polydet_cc_dcl_
#define _polydet_cc_dcl_

#include "./common.h"

__host__ __device__ void find_polynomial_determinant (
   PolyMatrix &Q, 
   PolyDegree deg, 
   int rows[Nrows], // This keeps the order of rows pivoted on. 
   int dim     // Actual dimension of the matrix
   );

#endif
