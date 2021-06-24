#ifndef _HIDDEN_6_H_
#define _HIDDEN_6_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef BUILD_MEX
#   include <mex.h>
#   define printf mexPrintf
#endif

// Dimensions of the matrices that we will be using
const int Nrows = 10;
const int Ncols = 10;
const int Maxdegree = 20; // Big enough for 5 or 6 point problem

// Basic Data Types
typedef double Matches[][3];
typedef double Matches_5[5][3];
typedef double Matches_6[6][3];
template <int n>
using Matches_n = double[n][3];

typedef double Ematrix[3][3];
typedef double Pmatrix[3][4];

typedef double EquationSet[5][10][10];

// For holding polynomials of matrices
typedef double Polynomial[Maxdegree+1];
typedef double PolyMatrix [Nrows][Ncols][Maxdegree+1];
typedef double PolyMatrix_p [Ncols][Maxdegree+1];
typedef int PolyDegree    [Nrows][Ncols];
typedef int PolyDegree_p [Ncols];

// We need to be able to solve matrix equations up to this dimension
typedef double BMatrix[Maxdegree+1][Maxdegree+1];

// Forward declarations
// void print_equation_set (EquationSet A, int maxdegree = 3);
void print_polymatrix   (PolyMatrix A, int maxdegree = 3);

__host__ __device__ void polyquotient (
         double *a, int sa, 
         double *b, double *t, int sb, 
         double *q, int &sq,
         BMatrix B, int &current_size
         );

__host__ __device__ void find_polynomial_determinant (
         PolyMatrix &Q,
         PolyDegree deg,
         int rows[Nrows],  // This keeps the order of rows pivoted on.
         int dim = Nrows
         );

__host__ __device__ void det_preprocess_6pt (
         PolyMatrix &Q, 
         PolyDegree degree, 
         int n_zero_roots  // Number of roots known to be zero
         );

__host__ __device__ void do_scale (
         PolyMatrix &Q, 
         PolyDegree degree, 
         double &scale_factor,   // Factor that x is multiplied by
         bool degree_by_row,
         int dim = Nrows
         );

__host__ __device__ inline double urandom ()
   {
   // Returns a real random between -1 and 1
   const int MAXRAND = 65000;
   return 2.0*((rand()%MAXRAND)/((double) MAXRAND) - 0.5);
   };

#endif
