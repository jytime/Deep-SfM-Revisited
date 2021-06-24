#include <string.h>
#include <stdio.h>

/* SFILE_BEGIN */
#include "common.h"
/* SFILE_END */

__host__ __device__ static void toeplitz_get_b_vector (
      BMatrix B, 
      double *t, int st,   
      int &current_size, int required_size
      )
   {
   // Incrementally computes the matrix of back-vectors used in the Levinson
   // algorithm. The back-vectors bn are stored as rows in B.

   // Initialize B
   if (current_size <= 0)
      {
      B[0][0] = 1.0/t[0];
      current_size = 1;
      }

   // Build up the back vectors one by one
   for (int n=current_size; n<required_size; n++)
      {
      // Fill out row n of the matrix B

      // Compute out each vector at once
      double e = 0.0;
      for (int i=1; i<st && i<=n; i++)
         e += t[i] * B[n-1][i-1];

      // Write the next row of the matrix
      double cb = 1.0 / (1.0 - e*e);
      double cf = -(e*cb);

      // Addresses into the arrays for the addition
      double *b0 = &B[n-1][0];
      double *f0 = &B[n-1][n];
      double *bn = &(B[n][0]);

      // First term does not include b, last does not have f
      *(bn++) = *(--f0) * cf;
      while (f0 != &B[n-1][0])
         *(bn++) = *(b0++) * cb + *(--f0) * cf;
      *bn = *b0 * cb;
      }

   // Update the current dimension
   current_size = required_size;
   }

__host__ __device__ void polyquotient (
   double *a, int sa, 
   double *b, double *t, int sb, 
   double *q, int &sq,
   BMatrix B, int &current_size
   )
   {
   // Computes the quotient of one polynomial with respect to another
   // in a least-squares sense, using Toeplitz matrices
   
   // First, get the sizes of the vectors
   sq = sa - sb + 1;  // Degree of the quotient

   // Next get the back-vectors for the Levinson algorithm
   if (sq > current_size)
      toeplitz_get_b_vector (B, t, sb, current_size, sq);

#ifdef RH_DEBUG
   for (int i=0; i<sq; i++)
      {
      for (int j=0; j<sq; j++)
          printf ("%9.3f ", B[i][j]);
      printf ("\n");
      }
#endif

   // Initially no values
   memset(q, 0, sq*sizeof(double));

   // Next, compute the quotient, one at a time
   for (int n=0; n<sq; n++)
      {
      // Inner product of a and b
      double yn = 0.0;
      for (int i=0; i<sb; i++)
         yn += b[i] * a[i+n];

      // The error value
      double e = 0.0;
      for (int i=1; i<sb && i<=n; i++)
         e += t[i] * q[n-i];

#ifdef RH_DEBUG
      printf ("yn = %12.6f, e = %12.6f\n", yn, e);
#endif

      // Now, update the value of q
      double fac = yn - e;
      q[n] = 0.0;
      for (int i=0; i<=n; i++)
         q[i] += fac * B[n][i];
      }
   }

