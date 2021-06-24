#ifndef _cheirality_cc_dcl_
#define _cheirality_cc_dcl_

__host__ __device__ void compute_P_matrices (
      Matches q, Matches qp, 
      Ematrix E[], 
      double *f,        // May be null
      Pmatrix P[], 
      int &nsolutions,
      int npoints  = 5 
   );

__host__ __device__ void compute_P_matrices_5pt (
      Matches q, Matches qp, 
      Pmatrix P[], 
      int &nsolutions,
      bool optimized
   );

#endif
