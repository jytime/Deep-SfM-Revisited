
// #include "Utilities_CC/utilities_CC.h"
// #include "Utilities_CC/Timer.h"
#include <math.h>

#ifdef WIN32
#include <float.h>
#   define isnan(x) _isnan(x)
#   define isinf(x) (! _finite(x))
#endif
#define notanum(x) (isnan(x) || isinf(x))

/* SFILE_BEGIN */
#include "essential_matrix_5pt_dcl.h"
#include "essential_matrix_5pt.h"
typedef double Matches[][3];
/* SFILE_END */

// Actual expected degree of the polynomial
const int PolynomialDegree = 10;

//=============================================================================
//           Various operators on the polynomial classes
//=============================================================================

__host__ __device__ poly4_2 poly4_1::operator * (poly4_1 p2)
   {
   poly4_1 &p1 = *this;
   poly4_2 prod;

   prod(0,0)  = p1(0)*p2(0);
   prod(0,1)  = p1(0)*p2(1);
   prod(0,2)  = p1(0)*p2(2);
   prod(0,3)  = p1(0)*p2(3);

   prod(0,1) += p1(1)*p2(0);
   prod(1,1)  = p1(1)*p2(1);
   prod(1,2)  = p1(1)*p2(2);
   prod(1,3)  = p1(1)*p2(3);

   prod(0,2) += p1(2)*p2(0);
   prod(1,2) += p1(2)*p2(1);
   prod(2,2)  = p1(2)*p2(2);
   prod(2,3)  = p1(2)*p2(3);

   prod(0,3) += p1(3)*p2(0);
   prod(1,3) += p1(3)*p2(1);
   prod(2,3) += p1(3)*p2(2);
   prod(3,3)  = p1(3)*p2(3);

   return prod;
   }

__host__ __device__ poly4_3 poly4_2::operator * (poly4_1 p2)
   {
   poly4_2 &p1 = *this;
   poly4_3 prod;

   prod(0,0,0)  = p1(0,0)*p2(0);
   prod(0,0,1)  = p1(0,0)*p2(1);
   prod(0,0,2)  = p1(0,0)*p2(2);
   prod(0,0,3)  = p1(0,0)*p2(3);

   prod(0,0,1) += p1(0,1)*p2(0);
   prod(0,1,1)  = p1(0,1)*p2(1);
   prod(0,1,2)  = p1(0,1)*p2(2);
   prod(0,1,3)  = p1(0,1)*p2(3);

   prod(0,0,2) += p1(0,2)*p2(0);
   prod(0,1,2) += p1(0,2)*p2(1);
   prod(0,2,2)  = p1(0,2)*p2(2);
   prod(0,2,3)  = p1(0,2)*p2(3);

   prod(0,0,3) += p1(0,3)*p2(0);
   prod(0,1,3) += p1(0,3)*p2(1);
   prod(0,2,3) += p1(0,3)*p2(2);
   prod(0,3,3)  = p1(0,3)*p2(3);

   prod(0,1,1) += p1(1,1)*p2(0);
   prod(1,1,1)  = p1(1,1)*p2(1);
   prod(1,1,2)  = p1(1,1)*p2(2);
   prod(1,1,3)  = p1(1,1)*p2(3);

   prod(0,1,2) += p1(1,2)*p2(0);
   prod(1,1,2) += p1(1,2)*p2(1);
   prod(1,2,2)  = p1(1,2)*p2(2);
   prod(1,2,3)  = p1(1,2)*p2(3);

   prod(0,1,3) += p1(1,3)*p2(0);
   prod(1,1,3) += p1(1,3)*p2(1);
   prod(1,2,3) += p1(1,3)*p2(2);
   prod(1,3,3)  = p1(1,3)*p2(3);

   prod(0,2,2) += p1(2,2)*p2(0);
   prod(1,2,2) += p1(2,2)*p2(1);
   prod(2,2,2)  = p1(2,2)*p2(2);
   prod(2,2,3)  = p1(2,2)*p2(3);

   prod(0,2,3) += p1(2,3)*p2(0);
   prod(1,2,3) += p1(2,3)*p2(1);
   prod(2,2,3) += p1(2,3)*p2(2);
   prod(2,3,3)  = p1(2,3)*p2(3);

   prod(0,3,3) += p1(3,3)*p2(0);
   prod(1,3,3) += p1(3,3)*p2(1);
   prod(2,3,3) += p1(3,3)*p2(2);
   prod(3,3,3)  = p1(3,3)*p2(3);

#ifdef RH_DEBUG
   printf ("In poly4_2 * poly4_1\n");
   printf ("poly4_2 = \n");
   p1.print();
   printf ("poly4_1 = \n");
   p2.print();
   printf ("poly4_2 * poly4_2 = \n");
   prod.print();
#endif

   return prod;
   }

__host__ __device__ poly4_3 poly4_3::operator * (double k)
   {
   poly4_3 &p1 = *this;
   poly4_3 prod;

   prod(0,0,0) = p1(0,0,0) * k;
   prod(0,0,1) = p1(0,0,1) * k;
   prod(0,0,2) = p1(0,0,2) * k;
   prod(0,0,3) = p1(0,0,3) * k;
   prod(0,1,1) = p1(0,1,1) * k;
   prod(0,1,2) = p1(0,1,2) * k;
   prod(0,1,3) = p1(0,1,3) * k;
   prod(0,2,2) = p1(0,2,2) * k;
   prod(0,2,3) = p1(0,2,3) * k;
   prod(0,3,3) = p1(0,3,3) * k;
   prod(1,1,1) = p1(1,1,1) * k;
   prod(1,1,2) = p1(1,1,2) * k;
   prod(1,1,3) = p1(1,1,3) * k;
   prod(1,2,2) = p1(1,2,2) * k;
   prod(1,2,3) = p1(1,2,3) * k;
   prod(1,3,3) = p1(1,3,3) * k;
   prod(2,2,2) = p1(2,2,2) * k;
   prod(2,2,3) = p1(2,2,3) * k;
   prod(2,3,3) = p1(2,3,3) * k;
   prod(3,3,3) = p1(3,3,3) * k;

   return prod;
   }

__host__ __device__ poly4_3 poly4_3::operator + (poly4_3 p2)
   {
   poly4_3 &p1 = *this;
   poly4_3 sum;

   sum(0,0,0) = p1(0,0,0) + p2(0,0,0);
   sum(0,0,1) = p1(0,0,1) + p2(0,0,1);
   sum(0,0,2) = p1(0,0,2) + p2(0,0,2);
   sum(0,0,3) = p1(0,0,3) + p2(0,0,3);
   sum(0,1,1) = p1(0,1,1) + p2(0,1,1);
   sum(0,1,2) = p1(0,1,2) + p2(0,1,2);
   sum(0,1,3) = p1(0,1,3) + p2(0,1,3);
   sum(0,2,2) = p1(0,2,2) + p2(0,2,2);
   sum(0,2,3) = p1(0,2,3) + p2(0,2,3);
   sum(0,3,3) = p1(0,3,3) + p2(0,3,3);
   sum(1,1,1) = p1(1,1,1) + p2(1,1,1);
   sum(1,1,2) = p1(1,1,2) + p2(1,1,2);
   sum(1,1,3) = p1(1,1,3) + p2(1,1,3);
   sum(1,2,2) = p1(1,2,2) + p2(1,2,2);
   sum(1,2,3) = p1(1,2,3) + p2(1,2,3);
   sum(1,3,3) = p1(1,3,3) + p2(1,3,3);
   sum(2,2,2) = p1(2,2,2) + p2(2,2,2);
   sum(2,2,3) = p1(2,2,3) + p2(2,2,3);
   sum(2,3,3) = p1(2,3,3) + p2(2,3,3);
   sum(3,3,3) = p1(3,3,3) + p2(3,3,3);

   return sum;
   }

__host__ __device__ void poly4_3::operator += (poly4_3 p2)
   {
   poly4_3 &p1 = *this;

   p1(0,0,0) += p2(0,0,0);
   p1(0,0,1) += p2(0,0,1);
   p1(0,0,2) += p2(0,0,2);
   p1(0,0,3) += p2(0,0,3);
   p1(0,1,1) += p2(0,1,1);
   p1(0,1,2) += p2(0,1,2);
   p1(0,1,3) += p2(0,1,3);
   p1(0,2,2) += p2(0,2,2);
   p1(0,2,3) += p2(0,2,3);
   p1(0,3,3) += p2(0,3,3);
   p1(1,1,1) += p2(1,1,1);
   p1(1,1,2) += p2(1,1,2);
   p1(1,1,3) += p2(1,1,3);
   p1(1,2,2) += p2(1,2,2);
   p1(1,2,3) += p2(1,2,3);
   p1(1,3,3) += p2(1,3,3);
   p1(2,2,2) += p2(2,2,2);
   p1(2,2,3) += p2(2,2,3);
   p1(2,3,3) += p2(2,3,3);
   p1(3,3,3) += p2(3,3,3);
   }

__host__ __device__ poly4_3 poly4_3::operator - (poly4_3 p2)
   {
   poly4_3 &p1 = *this;
   poly4_3 dif;

   dif(0,0,0) = p1(0,0,0) - p2(0,0,0);
   dif(0,0,1) = p1(0,0,1) - p2(0,0,1);
   dif(0,0,2) = p1(0,0,2) - p2(0,0,2);
   dif(0,0,3) = p1(0,0,3) - p2(0,0,3);
   dif(0,1,1) = p1(0,1,1) - p2(0,1,1);
   dif(0,1,2) = p1(0,1,2) - p2(0,1,2);
   dif(0,1,3) = p1(0,1,3) - p2(0,1,3);
   dif(0,2,2) = p1(0,2,2) - p2(0,2,2);
   dif(0,2,3) = p1(0,2,3) - p2(0,2,3);
   dif(0,3,3) = p1(0,3,3) - p2(0,3,3);
   dif(1,1,1) = p1(1,1,1) - p2(1,1,1);
   dif(1,1,2) = p1(1,1,2) - p2(1,1,2);
   dif(1,1,3) = p1(1,1,3) - p2(1,1,3);
   dif(1,2,2) = p1(1,2,2) - p2(1,2,2);
   dif(1,2,3) = p1(1,2,3) - p2(1,2,3);
   dif(1,3,3) = p1(1,3,3) - p2(1,3,3);
   dif(2,2,2) = p1(2,2,2) - p2(2,2,2);
   dif(2,2,3) = p1(2,2,3) - p2(2,2,3);
   dif(2,3,3) = p1(2,3,3) - p2(2,3,3);
   dif(3,3,3) = p1(3,3,3) - p2(3,3,3);

   return dif;
   }

__host__ __device__ poly4_2 poly4_2::operator + (poly4_2 p2)
   {
   poly4_2 &p1 = *this;
   poly4_2 sum;

   sum(0,0) = p1(0,0) + p2(0,0);
   sum(0,1) = p1(0,1) + p2(0,1);
   sum(0,2) = p1(0,2) + p2(0,2);
   sum(0,3) = p1(0,3) + p2(0,3);
   sum(1,1) = p1(1,1) + p2(1,1);
   sum(1,2) = p1(1,2) + p2(1,2);
   sum(1,3) = p1(1,3) + p2(1,3);
   sum(2,2) = p1(2,2) + p2(2,2);
   sum(2,3) = p1(2,3) + p2(2,3);
   sum(3,3) = p1(3,3) + p2(3,3);

   return sum;
   }

__host__ __device__ void poly4_2::operator += (poly4_2 p2)
   {
   poly4_2 &p1 = *this;

   p1(0,0) += p2(0,0);
   p1(0,1) += p2(0,1);
   p1(0,2) += p2(0,2);
   p1(0,3) += p2(0,3);
   p1(1,1) += p2(1,1);
   p1(1,2) += p2(1,2);
   p1(1,3) += p2(1,3);
   p1(2,2) += p2(2,2);
   p1(2,3) += p2(2,3);
   p1(3,3) += p2(3,3);
   }

__host__ __device__ poly4_2 poly4_2::operator - (poly4_2 p2)
   {
   poly4_2 &p1 = *this;
   poly4_2 dif;

   dif(0,0) = p1(0,0) - p2(0,0);
   dif(0,1) = p1(0,1) - p2(0,1);
   dif(0,2) = p1(0,2) - p2(0,2);
   dif(0,3) = p1(0,3) - p2(0,3);
   dif(1,1) = p1(1,1) - p2(1,1);
   dif(1,2) = p1(1,2) - p2(1,2);
   dif(1,3) = p1(1,3) - p2(1,3);
   dif(2,2) = p1(2,2) - p2(2,2);
   dif(2,3) = p1(2,3) - p2(2,3);
   dif(3,3) = p1(3,3) - p2(3,3);

   return dif;
   }

__host__ __device__ poly4_1 poly4_1::operator + (poly4_1 p2)
   {
   poly4_1 &p1 = *this;
   poly4_1 sum;

   sum(0) = p1(0) + p2(0);
   sum(1) = p1(1) + p2(1);
   sum(2) = p1(2) + p2(2);
   sum(3) = p1(3) + p2(3);

   return sum;
   }

__host__ __device__ poly4_1 poly4_1::operator - (poly4_1 p2)
   {
   poly4_1 &p1 = *this;
   poly4_1 dif;

   dif(0) = p1(0) - p2(0);
   dif(1) = p1(1) - p2(1);
   dif(2) = p1(2) - p2(2);
   dif(3) = p1(3) - p2(3);

   return dif;
   }

//=============================================================================

__host__ __device__ poly4_3 polydet4 (EmatrixSet_5pt E)
   {
   // Takes the determinant of a polynomial
   poly4_3 det = 
      (E(1,1)*E(2,2) - E(2,1)*E(1,2)) * E(0,0) +
      (E(2,1)*E(0,2) - E(0,1)*E(2,2)) * E(1,0) +
      (E(0,1)*E(1,2) - E(1,1)*E(0,2)) * E(2,0);

#ifdef RH_DEBUG
   printf ("Det =\n");
   det.print();
#endif

   return det;
   }

#define FULL_TRACE
#ifdef  FULL_TRACE
__host__ __device__ poly4_2 traceEEt (EmatrixSet_5pt E)
   {
   // Takes the trace of E E' -- returns a quadratic polynomial
   // Trace of product is the elementwise product of the elements

   poly4_2 tr = E(0,0) * E(0, 0) + E(0,1) * E(0, 1) + E(0,2) * E(0, 2) 
            + E(1,0) * E(1, 0) + E(1,1) * E(1, 1) + E(1,2) * E(1, 2) 
            + E(2,0) * E(2, 0) + E(2,1) * E(2, 1) + E(2,2) * E(2, 2);

#ifdef RH_DEBUG
   printf ("Trace is:\n");
   tr.print();
#endif

   return tr;
   }

#else

__host__ __device__ poly4_2 traceEEt (EmatrixSet_5pt E)
   {
   // We know that the trace has a simple form, provided that the
   // E-matrix basis is orthogonal.

   poly4_2 tr;  
   tr.clear();
   tr(0,0) = 1.0;
   tr(1,1) = 1.0;
   tr(2,2) = 1.0;
   tr(3,3) = 1.0;

   return tr;
   }

#endif

__host__ __device__ void mono_coeff (poly4_3 B, EquationSet A, int n)
   {
   // Extracts the monomial coefficients in x and y (with z = 1) from
   // a cubic homogeneous polynomial. Returns 4 vectors (degrees 0 to 3 in w)

   // Make some constants to make the code easier to read

   // Degrees of terms in w
   const int w0 = 0;
   const int w1 = 1;
   const int w2 = 2;
   const int w3 = 3;

   // Linear variables
   const int w = 0;
   const int x = 1;
   const int y = 2;
   const int z = 3;

   // Monomials
   const int xx  = 3;
   const int xy  = 4;
   const int yy  = 5;
   const int xxx = 6;
   const int xxy = 7;
   const int xyy = 8;
   const int yyy = 9;

   // Terms in w^0
   A[w0][n][ 0  ] = B(z, z, z);
   A[w0][n][ x  ] = B(x, z, z);
   A[w0][n][ y  ] = B(y, z, z);
   A[w0][n][ xx ] = B(x, x, z);
   A[w0][n][ yy ] = B(y, y, z);
   A[w0][n][ xy ] = B(x, y, z);
   A[w0][n][ xxx] = B(x, x, x);
   A[w0][n][ xxy] = B(x, x, y);
   A[w0][n][ xyy] = B(x, y, y);
   A[w0][n][ yyy] = B(y, y, y);

   // Terms in w^1
   A[w1][n][ 0  ] = B(w, z, z);
   A[w1][n][ x  ] = B(w, x, z);
   A[w1][n][ y  ] = B(w, y, z);
   A[w1][n][ xx ] = B(w, x, x);
   A[w1][n][ yy ] = B(w, y, y);
   A[w1][n][ xy ] = B(w, x, y);

   // Terms in w^2
   A[w2][n][ 0  ] = B(w, w, z);
   A[w2][n][ x  ] = B(w, w, x);
   A[w2][n][ y  ] = B(w, w, y);

   // Terms in w^3
   A[w3][n][ 0  ] = B(w, w, w);
   }

__host__ __device__ void EEeqns_5pt (EmatrixSet_5pt E, EquationSet A)
   {
   //
   // Computes the equations that will be used to input to polyeig.
   //    void EEeqns_5pt(E, A)
   // where E has dimensions E(3, 3, 4).  The output is a matrix
   // of dimension A(4, 10, 10), where A(i, :, :) is the coeffient of w^{i-1}
   //

   // Makes all the equations from the essential matrix E
   // First of all, set the equations to zero
   memset (&(A[0][0][0]), 0, sizeof(EquationSet));
   // Find the trace - this is a quadratic polynomial
   poly4_2 tr = traceEEt(E);
   // First equation is from the determinant
   mono_coeff (polydet4(E), A, 0);
   // Other equations from the equation 2 E*E'*E - tr(E*E') E = 0
   // In the following loop, we compute EE'E(i,j) = sum_pq E(i,p)*E(q,p)*E(q,j)
   // The way this is done is optimized for speed.  We compute first the matrix
   // EE'(i, q) and then use this to accumulate EE'E(i, j)

   int eqn = 1;  // Count on the next equation
   for (int i=0; i<3; i++)
      {
      // An array of cubic polynomials, one for each j = 0 ... 2
      poly4_3 EEE_i[3];  // Will hold (EE'E)(i,j)
      for (int j=0; j<3; j++) EEE_i[j].clear();

      // Compute each EE'(i,q) = sum_p E(i,p) E(q,p)
      for (int q=0; q<3; q++)
         {
         // Accumulate EE(i, q)
         poly4_2 EE_iq; EE_iq.clear();
         for (int p=0; p<3; p++)
            EE_iq += E(i,p) * E(q,p);

         // Now, accumulate EEE(ij) = sum_q  EE'(i,q) * E(q, j)
         for (int j=0; j<3; j++)
            EEE_i[j] += EE_iq * E(q,j);
         }

      // Now, EE'E(i,j) is computed for this i and all j
      // We can complete the computation of the coefficients from EE'E(i, j)
      for (int j=0; j<3; j++)
         mono_coeff(EEE_i[j]*2.0 - tr* E(i,j), A, eqn++);
      }
   }

__host__ __device__ void null_space_solve_3x3_half_pivot (double A[3][3], double &x, double &y)
   {
   //
   // Solve for the null-space of the matrix.  The value returned is
   // (x, y), where (1, x, y) is the generator of the null-space.
   //

   // This time we will do pivoting
   int p1;
   double f0 = fabs(A[0][2]), f1 = fabs(A[1][2]), f2 = fabs(A[2][2]);
   if (f0 > f1) p1 = (f0>f2)? 0 : 2;
   else p1 = (f1>f2) ? 1 : 2;

   // The other two rows
   int r1 = (p1+1)%3, r2 = (p1+2)%3;

   // Now, use this to pivot
   double fac = A[r1][2] / A[p1][2];
   A[r1][0] -= fac * A[p1][0];
   A[r1][1] -= fac * A[p1][1];

   fac = A[r2][2] / A[p1][2];
   A[r2][0] -= fac * A[p1][0];
   A[r2][1] -= fac * A[p1][1];
  
   // Second pivot - largest element in column 1
   int p2 = fabs(A[r1][1]) > fabs(A[r2][1]) ? r1 : r2;
   
   // Now, read off the values - back substitution
   x = - A[p2][0]               / A[p2][1];
   y = -(A[p1][0] + A[p1][1]*x) / A[p1][2];
   }

#if 0
//
// THIS DOES NOT COMPILE BECAUSE OF THE USE OF rhMatrix.
// IT IS POSSIBLE THAT IT SHOULD BE USED>
//
__host__ __device__ void null_space_solve_3x3 (double A[3][3], double &x, double &y)
   {
   // Solve for the null-space of the matrix
   rhMatrix AA (3, 3, &(A[0][0]));
   rhVector D(3);
   rhMatrix V(3, 3);

   // Take the SVD
   svd(AA, D, V);

   if (V[0][2] == 0.0)
      {
      // Just something that stops it from crashing
      x = 0.0;
      y = 0.0;
      }
   else
      {
      x = V[1][2] / V[0][2];
      y = V[2][2] / V[0][2];
      }
   }
#endif

__host__ __device__ void null_space_solve_5x9_nopivot (double A[5][9], EmatrixSet_5pt &E)
   {
   // This will compute the set of solutions for the equations
   // Sweep out one column at a time, starting with highest column number

   // We do Gaussian elimination to convert M to the form M = [X | I]
   // Then the null space will be [-I | X].

   // For present, this is done without pivoting.  
   // Mostly, do not need to actually change right hand part (that becomes I)

   const int lastrow  = 4;
   const int firstcol = 4; // First column to do elimination to make I
   const int lastcol  = 8; 

   // First sweep is to get rid of the above diagonal parts
   for (int col=lastcol; col>firstcol; col--)  // No need to do first col
      {
      // Remove column col
      const int row = col-firstcol; // Row to pivot around
      const double pivot = A[row][col];

      // Sweep out all rows up to the current one 
      for (int i=0; i<row; i++)
         {
         // This factor of the pivot row is to subtract from row i
         const double fac = A[i][col] / pivot;

         // Constant terms
         for (int j=0; j<col; j++)
            A[i][j] -= fac * A[row][j];
         }
      }

   // Now, do backward sweep to clear below the diagonal
   for (int col=firstcol; col<lastcol; col++) // No need to do lastcol
      {
      // Remove column col
      const int row = col-firstcol; // Row to pivot around
      const double pivot = A[row][col];

      // Sweep out all rows up to the current one 
      for (int i=row+1; i<=lastrow; i++)
         {
         // This factor of the pivot row is to subtract from row i
         const double fac = A[i][col] / pivot;

         // Constant terms
         for (int j=0; j<firstcol; j++)
            A[i][j] -= fac * A[row][j];
         }
      }

   // Make this into a matrix of solutions
   double fac;
   E(0, 0) = poly4_1(1.0, 0.0, 0.0, 0.0);
   E(0, 1) = poly4_1(0.0, 1.0, 0.0, 0.0);
   E(0, 2) = poly4_1(0.0, 0.0, 1.0, 0.0);
   E(1, 0) = poly4_1(0.0, 0.0, 0.0, 1.0);
   fac = -1.0/A[0][4];
   E(1, 1) = poly4_1(fac*A[0][0], fac*A[0][1], fac*A[0][2], fac*A[0][3]);
   fac = -1.0/A[1][5];
   E(1, 2) = poly4_1(fac*A[1][0], fac*A[1][1], fac*A[1][2], fac*A[1][3]);
   fac = -1.0/A[2][6];
   E(2, 0) = poly4_1(fac*A[2][0], fac*A[2][1], fac*A[2][2], fac*A[2][3]);
   fac = -1.0/A[3][7];
   E(2, 1) = poly4_1(fac*A[3][0], fac*A[3][1], fac*A[3][2], fac*A[3][3]);
   fac = -1.0/A[4][8];
   E(2, 2) = poly4_1(fac*A[4][0], fac*A[4][1], fac*A[4][2], fac*A[4][3]);

// #define USE_TEST_VALUES
#ifdef  USE_TEST_VALUES

   // Put an artificial value in 
   E(0,0)(0) =  2; E(0,1)(0) =   4; E(0,2)(0) = -1;
   E(1,0)(0) =  4; E(1,1)(0) =   5; E(1,2)(0) = -8;
   E(2,0)(0) =  2; E(2,1)(0) = -11; E(2,2)(0) =  8;

   E(0,0)(1) =  0; E(0,1)(1) =  -1; E(0,2)(1) =  2;
   E(1,0)(1) =  1; E(1,1)(1) =   7; E(1,2)(1) =  1;
   E(2,0)(1) = -2; E(2,1)(1) =   6; E(2,2)(1) =  7;

   E(0,0)(2) =  2; E(0,1)(2) =  -3; E(0,2)(2) =  7;
   E(1,0)(2) =  1; E(1,1)(2) =  -3; E(1,2)(2) = -9;
   E(2,0)(2) =  4; E(2,1)(2) =   1; E(2,2)(2) = -9;

   E(0,0)(3) =  5; E(0,1)(3) =   2; E(0,2)(3) =  7;
   E(1,0)(3) =  1; E(1,1)(3) =  -2; E(1,2)(3) = -4;
   E(2,0)(3) =  5; E(2,1)(3) =  -1; E(2,2)(3) =  8;

#endif
   }

__host__ __device__ void null_space_solve_5x9 (double A[9][9], EmatrixSet_5pt &E)
   {
   // This will compute the set of solutions for the equations
   // We do orthogonal reduction of the rows of the matrix
   const int nrows = 9;
   const int ncols = 9;

   // First, fill out some random entries in the remaining rows
   const double PPi = 3.18730379;       // Basically a random number
   double ran = PPi;
   for (int i=5; i<nrows; i++)
      {
      for (int j=0; j<ncols; j++)
         {
         ran *= PPi;
         ran = 2.0 * (ran - floor(ran)) - 1.0;
         A[i][j] = ran;
         }
      }
   // Now, do Gram-Schmidt
   for (int row=0; row<nrows; row++)
      {
      // Normalize the row
      double sum = 0.0;
      for (int j=0; j<ncols; j++) sum += A[row][j]*A[row][j];
      double fac = 1.0 / sqrt(sum);
      for (int j=0; j<ncols; j++) A[row][j] *= fac;

      // Use to sweep out the subsequent rows
      for (int i=row+1; i<nrows; i++)
         {
         // Inner product of row i and row j
         double prod = 0.0;
         for (int j=0; j<ncols; j++)
            prod += A[row][j]*A[i][j]; // Inner product
         for (int j=0; j<ncols; j++) A[i][j] -= prod * A[row][j];
         }
      }

   // Make this into a matrix of solutions
   int count = 0;
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         {
         E(i,j) = poly4_1(A[5][count], A[6][count], A[7][count], A[8][count]);
         count++;
         }
   }

__host__ __device__ void Ematrix_5pt(Matches q, Matches qp, EmatrixSet_5pt &E, EquationSet &A)
   {
   // Computes the E-matrix from match inputs
   // A matrix to solve linearly for the ematrix
   double M[9][9];
   memset (&(M[0][0]), 0, sizeof (M));

   for (int i=0; i<5; i++)
      {
      M[i][0] = qp[i][0]*q[i][0];
      M[i][1] = qp[i][0]*q[i][1]; 
      M[i][2] = qp[i][0]*q[i][2];
      M[i][3] = qp[i][1]*q[i][0];
      M[i][4] = qp[i][1]*q[i][1]; 
      M[i][5] = qp[i][1]*q[i][2]; 
      M[i][6] = qp[i][2]*q[i][0];
      M[i][7] = qp[i][2]*q[i][1];
      M[i][8] = qp[i][2]*q[i][2]; 
      }
   // Solve using null_space_solve
   null_space_solve_5x9 (M, E);
#  ifdef RH_DEBUG
      printf ("E = \n");
      E.print();
#  endif

   // Now, get the equations
   EEeqns_5pt(E, A);
#  ifdef RH_DEBUG
   print_equation_set (A, 3); 
#endif
   }

__host__ __device__ void sweep_up (EquationSet A, int row, int col, int degree)
   {
   // Use the given pivot point to sweep out above the pivot
   const int num1 = 6; // number of nonzero columns of A in degree 1
   const int num2 = 3; // number of nonzero columns of A in degree 2
   const int num3 = 1; // number of nonzero columns of A in degree 3

   // Find the pivot value
   const double pivot = A[degree][row][col];

   // Sweep out all rows up to the current one 
   for (int i=0; i<row; i++)
      {
      // This factor of the pivot row is to subtract from row i
      const double fac = A[degree][i][col] / pivot;

      // Constant terms
      for (int j=0; j<=col; j++)
         A[0][i][j] -= fac * A[0][row][j];

      // Degree 1 terms
      for (int j=0; j<num1; j++)
         A[1][i][j] -= fac * A[1][row][j];

      // Degree 2 terms
      for (int j=0; j<num2; j++)
         A[2][i][j] -= fac * A[2][row][j];

      // Degree 3 terms
      for (int j=0; j<num3; j++)
         A[3][i][j] -= fac * A[3][row][j];
      }
   }

__host__ __device__ void sweep_down (EquationSet A, int row, int col, int degree, int lastrow)
   {
   // Use the given pivot point to sweep out below the pivot
   const int num1 = 6; // number of nonzero columns of A in degree 1
   const int num2 = 3; // number of nonzero columns of A in degree 2
   const int num3 = 1; // number of nonzero columns of A in degree 3

   // The value of the pivot point
   const double pivot = A[degree][row][col];

   // Sweep out all rows up to the current one 
   for (int i=row+1; i<=lastrow; i++)
      {
      // This factor of the pivot row is to subtract from row i
      const double fac = A[degree][i][col] / pivot;

      // Constant terms
      for (int j=0; j<=col; j++)
         A[0][i][j] -= fac * A[0][row][j];

      // Degree 1 terms
      for (int j=0; j<num1; j++)
         A[1][i][j] -= fac * A[1][row][j];

      // Degree 2 terms
      for (int j=0; j<num2; j++)
         A[2][i][j] -= fac * A[2][row][j];

      // Degree 3 terms
      for (int j=0; j<num3; j++)
         A[3][i][j] -= fac * A[3][row][j];
      }
   }

__host__ __device__ void print_equation_set (EquationSet A, int maxdegree)
   {
   // Print out the matrix
   printf ("Equation matrix\n");
   for (int degree=0; degree<=maxdegree; degree++)
      {
      for (int i=0; i<10; i++)
         {
         for (int j=0; j<10; j++)
            printf ("%7.1f ", A[degree][i][j]);
         printf ("\n");
         }
      printf ("\n");
      } 
   }

__host__ __device__ void print_Ematrix (Ematrix E)
   {
   // Not actually used in this directory at least.
   for (int i=0; i<3; i++)
      printf ("\t%12.5f  %12.5f  %12.5f\n", E[i][0], E[i][1], E[i][2]);
   printf ("\n");
   }

__host__ __device__ inline void swap (double &a, double &b)
   {
   double temp = a; a = b; b = temp;
   }

__host__ __device__ void pivot (EquationSet A, int last, int deg, int dummy)
   {
   // Pivot so that the largest element in the column is in the diagonal

   // Use the given pivot point to sweep out below the pivot
   const int num1 = 6; // number of nonzero columns of A in degree 1
   const int num2 = 3; // number of nonzero columns of A in degree 2
   const int num3 = 1; // number of nonzero columns of A in degree 3

   // Find the maximum value in the column
   double maxval = fabs(A[deg][last][last]);
   int row = last;
   for (int i=0; i<last; i++)
      {
      if (fabs(A[deg][i][last]) > maxval)
         {
         row = i;
         maxval = fabs(A[deg][i][last]);
         }
      }

   // If is in the maximum position, then return.
   if (row == last) return;

   // Otherwise, swap
   // Constant terms
   for (int j=0; j<=last; j++)
      swap(A[0][last][j], A[0][row][j]);

   // Degree 1 terms
   for (int j=0; j<num1; j++)
      swap(A[1][last][j], A[1][row][j]);

   // Degree 2 terms
   for (int j=0; j<num2; j++)
      swap(A[2][last][j], A[2][row][j]);

   // Degree 3 terms
   for (int j=0; j<num3; j++)
      swap(A[3][last][j], A[3][row][j]);
   }

__host__ __device__ void reduce_Ematrix (EquationSet A)
   {
   // This reduces the equation set to 3 x 3.  In this version there is
   // no pivoting, which relies on the pivots to be non-zero.

   // Relies on the particular form of the A matrix to reduce it
   // That means that there are several rows of zero elements in different
   // degrees, as given below.

   // Sweeping out the constant terms to reduce to 6 x 6
   pivot (A, 9, 0, 8); sweep_up (A, 9, 9, 0);
   pivot (A, 8, 0, 7); sweep_up (A, 8, 8, 0);
   pivot (A, 7, 0, 6); sweep_up (A, 7, 7, 0);
   pivot (A, 6, 0, 5); sweep_up (A, 6, 6, 0);

   // Now, the matrix is 6 x 6.  Next we need to handle linear terms
   pivot (A, 5, 0, 4); sweep_up (A, 5, 5, 0);
   pivot (A, 4, 0, 3); sweep_up (A, 4, 4, 0);
   pivot (A, 3, 0, 2); sweep_up (A, 3, 3, 0);

   int lastrow = 5;
   sweep_down (A, 3, 3, 0, lastrow);
   sweep_down (A, 4, 4, 0, lastrow);

   // Also sweep out the first-order terms
   sweep_up   (A, 2, 5, 1);
   sweep_up   (A, 1, 4, 1);

   sweep_down (A, 0, 3, 1, lastrow);
   sweep_down (A, 1, 4, 1, lastrow);
   sweep_down (A, 2, 5, 1, lastrow);

   // Now, sweep out the x terms by increasing the degree
   for (int i=0; i<3; i++)
      {
      double fac = A[1][i][3+i] / A[0][3+i][3+i];

      // Introduces 4-th degree term
      A[4][i][0] = -A[3][i+3][0] * fac;

      // Transfer terms of degree 0 to 3
      for (int j=0; j<3; j++)
         {
         A[3][i][j] -= A[2][i+3][j] * fac;
         A[2][i][j] -= A[1][i+3][j] * fac;
         A[1][i][j] -= A[0][i+3][j] * fac;
         }
      }
   }

__host__ __device__ void reduce_constant_terms (EquationSet A)
   {
   // This reduces the equation set to 6 x 6 by eliminating the
   // constant terms at the end.  In this
   // no pivoting, which relies on the pivots to be non-zero.

   // Sweeping out the constant terms to reduce to 6 x 6
   pivot (A, 9, 0, 8); sweep_up (A, 9, 9, 0);
   pivot (A, 8, 0, 7); sweep_up (A, 8, 8, 0);
   pivot (A, 7, 0, 6); sweep_up (A, 7, 7, 0);
   pivot (A, 6, 0, 5); sweep_up (A, 6, 6, 0);
   }

__host__ __device__ inline void one_cofactor (EquationSet A, Polynomial poly, 
   int r0, int r1, int r2)
   {
   // Computes one term of the 3x3 cofactor expansion

   // Get a polynomial to hold a 2x2 determinant
   double two[7];
   memset (&(two[0]), 0, 7*sizeof(double));

   // Compute the 2x2 determinant - results in a 6-degree polynomial
   for (int i=0; i<=3; i++)
      for (int j=0; j<=3; j++)
         two [i+j] += A[i][r1][1]*A[j][r2][2] - A[i][r2][1]*A[j][r1][2];

   // Now, multiply by degree 4 polynomial
   for (int i=0; i<=6; i++)
      for (int j=0; j<=4; j++)
         poly [i+j] += A[j][r0][0]*two[i];
   }

__host__ __device__ void compute_determinant (EquationSet A, Polynomial poly)
   {
   // Does the final determinant computation to return the determinant
   // Input is a 3x3 matrix of polynomialsm A,
   // Output is poly (degree 10)

   // Clear out the polynomial
   memset (&(poly[0]), 0, (PolynomialDegree+1)*sizeof(double));

   // Now, the three cofactors
   one_cofactor (A, poly, 0, 1, 2);
   one_cofactor (A, poly, 1, 2, 0);
   one_cofactor (A, poly, 2, 0, 1);
   }


// Declaration of the function to find roots
__host__ __device__ int find_real_roots_sturm( 
   double *p, int order, double *roots, int *nroots, bool non_neg = false);

__host__ __device__ void compute_E_matrix (EmatrixSet_5pt &Es, EquationSet &A, double w, Ematrix &E)
   {
   // Compute the essential matrix corresponding to this root
   double w2 = w*w;
   double w3 = w2*w;
   double w4 = w3*w;
 
   // Form equations to solve
   double M[3][3];
   for (int i=0; i<3; i++)
      {
      for (int j=0; j<3; j++)
         {
         M[i][j] = A[0][i][j] + w*A[1][i][j] + w2*A[2][i][j] + w3*A[3][i][j];
         }

      // Only the first row has degree 4 terms
      M[i][0] += w4*A[4][i][0];
      }

   // Now, find the solution
   double x, y;
   null_space_solve_3x3_half_pivot (M, x, y);

   //-----------------------------------------------------
   // Insurance that it worked
   // If this failed, then try again with different method

#if 0
   //
   // POSSIBLY THIS SHOULD BE USED, BUT IT DOES NOT COMPILE.
   //
   if (notanum(x) || notanum(y))
      {
      // Do it again
      for (int i=0; i<3; i++)
         {
         for (int j=0; j<3; j++)
            {
            M[i][j] = A[0][i][j] + w*A[1][i][j] + w2*A[2][i][j] + w3*A[3][i][j];
            }

         // Only the first row has degree 4 terms
         M[i][0] += w4*A[4][i][0];
         }

      // Solve using safer SVD solver
      null_space_solve_3x3 (M, x, y);
      }
#endif

   //-----------------------------------------------------

   // Multiply out the solution to get the essential matrix
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         {
         poly4_1 &p = Es(i, j);
         E[i][j] = w*p(0) + x*p(1) + y*p(2) + p(3);
         }
   }

__host__ __device__ void compute_E_A_poly (
        Matches q, Matches qp,
        double EE[4][3][3],
        double AA[5][3][3],
        Polynomial poly)
   {
   // This is used by the Matlab interface.
   // It takes the matches and returns the basis for the E-matrices (EE)
   // along with a 3x3 matrix of polynomials, which allows us to solve
   // for w.  It also returns the polynomial to solve

   // Get the matrix set
   EquationSet A;
   EmatrixSet_5pt E;
   Ematrix_5pt(q, qp, E, A);

   // Now, reduce its dimension to 3 x 3
   reduce_Ematrix (A);

   // Finally, get the 10-th degree polynomial out of this
   if (poly) compute_determinant (A, poly);

   // Now, copy to the simple arrays
   if (EE)
      for (int d=0; d<4; d++) for (int i=0; i<3; i++) for (int j=0; j<3; j++)
         EE[d][i][j] = E(i,j)(d);  // Do not transpose - we want Ematrices thus

   if (AA)
      for (int d=0; d<5; d++) for (int i=0; i<3; i++) for (int j=0; j<3; j++)
         AA[d][i][j] = A[d][j][i]; // Transpose
   }

__host__ __device__ static inline double pval (double *p, int deg, double x)
   {
   // Evaluates a polynomial at a given point x.  Assumes deg >= 0
   double val = p[deg];
   for (int i=deg-1; i>=0; i--)
      val = x*val + p[i];
   return val;
   }

__host__ __device__ static void compute_E_matrix_generic (
      EmatrixSet_5pt &Es,
      PolyMatrix A,
      PolyDegree deg,         // Degree of each entry in A
      int rows[Nrows],
      double w,
      double scale,
      Ematrix &E
      )
   {
   // Compute the essential matrix corresponding to this root from
   // the matrix of equations A, assumed to be in row-echelon form
   // as defined by the array rows.

   double a10 = pval(A[rows[1]][0], deg[rows[1]][0], w);
   double a11 = pval(A[rows[1]][1], deg[rows[1]][1], w);
   double a20 = pval(A[rows[2]][0], deg[rows[2]][0], w);
   double a21 = pval(A[rows[2]][1], deg[rows[2]][1], w);
   double a22 = pval(A[rows[2]][2], deg[rows[2]][2], w);

   double x = -a10/a11;
   double y = -(a20 + x*a21) / a22;

   // Multiply out the solution to get the essential matrix
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         {
         poly4_1 &p = Es(i, j);
         E[i][j] = scale*w*p(0) + x*p(1) + y*p(2) + p(3);
         }
   }

__host__ __device__ void compute_E_matrices (
     Matches q, Matches qp, 
     Ematrix Ematrices[10], 
     int &nroots,
     bool optimized
     )
   {
   // Get the matrix set
   EquationSet A;
   EmatrixSet_5pt E;
   Ematrix_5pt(q, qp, E, A);
   // print_equation_set (A, 3); 

   if (!optimized)
      {
      //------------------------------------------------------------------------
      // This is the generic version of the solver as in our paper
      //------------------------------------------------------------------------

      int dim = Nrows;

      // First of all, reduce to 6 x 6 by eliminating constant columns
      reduce_constant_terms (A);
      dim = 6;

      // Set up array of degrees
      PolyDegree degrees;
      for (int i=0; i<dim; i++)
         {
         degrees[i][0] = 3;
         degrees[i][1] = 2;
         degrees[i][2] = 2;
         degrees[i][3] = 1;
         degrees[i][4] = 1;
         degrees[i][5] = 1;
         degrees[i][6] = 0;
         degrees[i][7] = 0;
         degrees[i][8] = 0;
         degrees[i][9] = 0;
         }

      // Unfortunately, we need to rearrange the data since it is incompatible
      PolyMatrix P;
      for (int i=0; i<dim; i++)
         for (int j=0; j<dim; j++)
            for (int d=0; d<=degrees[i][j]; d++)
                P[i][j][d] = A[d][i][j];

      // print_polymatrix (P, 3);

      // Go ahead and find the polynomial determinant
      double scale_factor = 1.0;
      do_scale (P, degrees, scale_factor, false, dim);

      int rows[Nrows];
      find_polynomial_determinant (P, degrees, rows, dim);
      double *poly = P[rows[0]][0];
      int poly_degree = degrees[rows[0]][0];

      // Find the positive real roots
      double roots[Maxdegree];
      find_real_roots_sturm(poly, poly_degree, roots, &nroots);

      // Now, get the ematrices
      for (int i=0; i<nroots; i++)
         compute_E_matrix_generic (E, P, degrees, rows,
            roots[i], scale_factor, Ematrices[i]);
      }

   else
      {
      //------------------------------------------------------------------------
      // This is the highly optimized version of the code - similar to Nister's
      //------------------------------------------------------------------------

      // Now, reduce its dimension to 3 x 3
      reduce_Ematrix (A);

      // Finally, get the 10-th degree polynomial out of this
      Polynomial poly;
      compute_determinant (A, poly);

      // Find the roots
      double roots[PolynomialDegree];
      find_real_roots_sturm(poly, PolynomialDegree, roots, &nroots);

      // Now, get the ematrices
      for (int i=0; i<nroots; i++) 
         compute_E_matrix (E, A, roots[i], Ematrices[i]);
      }

// #define PRINT_RESULTS
#ifdef PRINT_RESULTS
#undef PRINT_RESULTS
   printf ("Polynomial\n");
   for (int i=0; i<=PolynomialDegree; i++)
      printf ("\t%14.6f\n", poly[i]/poly[0]);
#endif

// #define PRINT_RESULTS
#ifdef PRINT_RESULTS
#undef PRINT_RESULTS
   // Print out the roots
   printf ("Roots\n");
   for (int i=0; i<nroots; i++)
      printf ("\t%14.6f\n", roots[i]);
#endif

// #define PRINT_RESULTS
#ifdef PRINT_RESULTS
#undef PRINT_RESULTS
   // Print out the essential matrices
   printf ("Ematrices\n");
   for (int m=0; m<nroots; m++)
      {
      const Ematrix &E = Ematrices[m];
      for (int i=0; i<3; i++)
         printf ("\t%12.5f  %12.5f  %12.5f\n", E[i][0], E[i][1], E[i][2]);
      printf ("\n");

      // Now, compute to see if it has worked
      printf ("Verify: ");
      for (int pt=0; pt<5; pt++) 
         {
         double sum = 0.0;
         for (int i=0; i<3; i++) for (int j=0; j<3; j++)
            sum += qp[pt][i] * E[i][j] * q[pt][j];
         printf ("%11.3e ", sum);
         }
      printf ("\n\n");
      }
#endif
   }

__host__ __device__ void compute_E_matrices_optimized (
     Matches q, Matches qp,
     Ematrix Ematrices[10],
     int &nroots
     )
   {
   // Get the matrix set
   EquationSet A;
   EmatrixSet_5pt E;
   Ematrix_5pt(q, qp, E, A);

   // Now, reduce its dimension to 3 x 3
   reduce_Ematrix(A);

   // Finally, get the 10-th degree polynomial out of this
   Polynomial poly;
   compute_determinant(A, poly);

   // Find the roots
   double roots[PolynomialDegree];
   find_real_roots_sturm(poly, PolynomialDegree, roots, &nroots); // ~50MB

   // Now, get the ematrices
   for (int i=0; i<nroots; i++)
      compute_E_matrix(E, A, roots[i], Ematrices[i]);
   }

