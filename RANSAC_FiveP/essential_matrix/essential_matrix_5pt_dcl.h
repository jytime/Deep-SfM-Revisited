#ifndef _essential_matrix_5pt_cc_dcl_
#define _essential_matrix_5pt_cc_dcl_


#include "essential_matrix_5pt.h"
typedef double Matches[][3];

__host__ __device__ void print_Ematrix (Ematrix E);

__host__ __device__ void compute_E_matrices (
     Matches q, Matches qp, 
     Ematrix Ematrices[10], 
     int &nroots,
     bool optimized
     );

__host__ __device__ void compute_E_matrices_optimized (
     Matches q, Matches qp,
     Ematrix Ematrices[10],
     int &nroots
     );

#endif
