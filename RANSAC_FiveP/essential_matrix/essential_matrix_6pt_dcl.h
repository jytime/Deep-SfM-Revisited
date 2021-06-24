#ifndef _essential_matrix_6pt_cc_dcl_
#define _essential_matrix_6pt_cc_dcl_


#include "essential_matrix_6pt.h" 
typedef double Matches[][3];

__host__ __device__ void compute_F_matrices_6pt (
         Matches q, Matches qp, 
         Ematrix Ematrices[Maxdegree], 
         double *flengths,
         int &nroots);

__host__ __device__ void compute_E_matrices_6pt (
         Matches q, Matches qp, 
         Ematrix Ematrices[Maxdegree], 
         double *flengths,
         int &nroots);

#endif
