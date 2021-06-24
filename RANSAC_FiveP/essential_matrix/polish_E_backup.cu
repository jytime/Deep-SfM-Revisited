#include <stdio.h>
#include <math.h>
#include <string.h>

#include "common.h"
#include "polish_E.h"

// Some type definitions -- could get from common.h
// typedef double Matches_5[5][3];
// typedef double Ematrix[3][3];

//----------------------------------------------------------------------------
//
//  This relies on a parametrization for an Essential Matrix as given
//  by the decomposition: 
//
//       Rx . Ry . Rz . E . Ru' . Rv' = I2
//     
//  where Ru and Rv are rotations about the x and y axes.
//
//  From this it follows that
//
//       E = Rz' . Ry' . Rx' . I2 . Rv . Ru
//
//  We also define 
//
//       U = Rz' . Ry' . Rx'
//       V = Ru' . Rv'
//
//  Then E = U . I2 . V'
// 
//  The essential matrix satisfies  q' . E . p = 0.  So
//
//       q' . U . I2 . V' . p = 0.
//
//  We define pp = V' . p and qq = U' . q.  Then
//
//       qq' . I2 . pp = 0
//
//  More generally, if we define qq and pp as above, and find E0 such
//  that 
//       qq' . E0 . pp = 0,
//
//  then
//
//       q' . (U . E0 . V) . p = 0
//
//  Hence, if 
//
//       E0 = U0 . I2 . V0' 
//          = Rz' . Ry' . Rx' . I2 . Rv . Ru
//
//  then, the update step is to replace
//
//       U -> U . Rz'. Ry' . Rx'
//       V -> V . Ru' . Rv'
//
//   We find that if qq' . I2 . pp = delta
//   then, we want to solve
//   
//          qq' .  Rz' . Ry' . Rx' . I2 . Rv . Ru . pp = 0
//
//   To do this, we take derivatives about the origin (z = y = x = u = v = 0)
//
//   We have to solve J . [dx, dy, dz, dv, du] = -delta
//
//   Canonical order for the 5 rotations is x, y, z, v, u, which is the
//   order that they are applied to E in extracting the parameters.
//   Namely E = Rz . Ry . Rx . I2 . Rv' . Ru'.
//
//----------------------------------------------------------------------------

__host__ __device__ void Eprod (Ematrix U, Ematrix V, Ematrix E)
   {
   // Forms E = U . I2 . V'
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         E[i][j] = U[i][0]*V[j][0] + U[i][1]*V[j][1];
   }

__host__ __device__ void printE  (Ematrix E)
   {
   for (int i=0; i<3; i++)
      {
      for (int j=0; j<3; j++)
         printf (" %20.16f", E[i][j]);
      printf ("\n");
      }

   printf ("\n");
   }

__host__ __device__ void printM  (Matches_5 E)
   {
   for (int i=0; i<5; i++)
      {
      for (int j=0; j<3; j++)
         printf (" %20.16f", E[i][j]);
      printf ("\n");
      }

   printf ("\n");
   }

__host__ __device__ void printA  (double A[5][5])
   {
   for (int i=0; i<5; i++)
      {
      for (int j=0; j<5; j++)
         printf (" %8.4f", A[i][j]);
      printf ("\n");
      }

   printf ("\n");
   }

__host__ __device__ void printAb  (double A[5][5], double b[5])
   {
   for (int i=0; i<5; i++)
      {
      for (int j=0; j<5; j++)
         printf (" %8.4f", A[i][j]);
      printf (" | %8.4f\n", b[i]);
      }

   printf ("\n");
   }

__host__ __device__ inline void Gright (Ematrix E, int row1, int row2, double angle)
   {
   // Applies a Givens rotation to a matrix from the right
   // Multiplies A on the right by G';  E -> E . G'

   // Get cos and sin of angle
   double c = cos(angle);
   double s = sin(angle);
   
   // Now, carry out
   for (int i=0; i<3; i++)
      {
      double temp = E[i][row1] * c - E[i][row2] * s;
      E[i][row2]  = E[i][row1] * s + E[i][row2] * c;
      E[i][row1]  = temp;
      }
   }

__host__ __device__ void Edecomp(Ematrix E, Ematrix U, Ematrix V)
   {
   //
   // Given an essential matrix E,
   // computes rotation matrices U and V such that E = U . I2 . V'
   // where I2 = diag(1, 1, 0)
   //

   // Parameters of the matrix
   double cx, cy, cz, cu, cv, sx, sy, sz, su, sv, temp, scale;

   // printE(E);

   //----------------
   // Gleft (0, 1, 0)
   cz =  E[0][0];
   sz = -E[1][0]; // Element to be eliminated
   scale = sqrt(cz*cz + sz*sz);
   cz /= scale;
   sz /= scale;
   
   // Now, carry out
   for (int j=0; j<3; j++)
      {
      temp     = E[0][j] * cz - E[1][j] * sz;
      E[1][j]  = E[0][j] * sz + E[1][j] * cz;
      E[0][j]  = temp;
      }

   // printE(E);

   //----------------
   // Gleft (0, 2, 0)
   cy =  E[0][0];
   sy = -E[2][0]; // Element to be eliminated
   scale = sqrt(cy*cy + sy*sy);
   cy /= scale;
   sy /= scale;
   
   // Now, carry out
   for (int j=0; j<3; j++)
      {
      temp     = E[0][j] * cy - E[2][j] * sy;
      E[2][j]  = E[0][j] * sy + E[2][j] * cy;
      E[0][j]  = temp;
      }

   // printE(E);

   //----------------
   // Gleft (1, 2, 1)
   cx =  E[1][1];
   sx = -E[2][1]; // Element to be eliminated
   scale = sqrt(cx*cx + sx*sx);
   cx /= scale;
   sx /= scale;
   
   // Now, carry out -- no need to compute row 2
   for (int j=1; j<3; j++)
      E[1][j]  = E[1][j] * cx - E[2][j] * sx;

   // printE(E);

   // Compute the right matrix
   U[0][0] =  cy*cz; U[0][1] = -cz*sx*sy + cx*sz; U[0][2] = cx*cz*sy + sx*sz;
   U[1][0] = -cy*sz; U[1][1] =  cx*cz + sx*sy*sz; U[1][2] = cz*sx - cx*sy*sz;
   U[2][0] =    -sy; U[2][1] =            -cy*sx; U[2][2] =            cx*cy;

   //-----------------------
   // Now, column operations
   // Gright(1, 2, 1)
   cu =  E[1][1];
   su = -E[1][2]; // Element to be eliminated
   scale = sqrt(cu*cu + su*su);
   cu /= scale;
   su /= scale;
   
   // Now, carry out
   E[0][2]  = su * E[0][1] + cu * E[0][2];

   // printE(E);

   // Gright(0, 2, 0)
   cv =  E[0][0];
   sv = -E[0][2]; // Element to be eliminated
   scale = sqrt(cv*cv + sv*sv);
   cv /= scale;
   sv /= scale;

   // printE(E);

   // Compute the right matrix
   V[0][0] =     cv; V[0][1] =   0; V[0][2] =    sv;
   V[1][0] = -su*sv; V[1][1] =  cu; V[1][2] = cv*su;
   V[2][0] = -cu*sv; V[2][1] = -su; V[2][2] = cu*cv;
   }

__host__ __device__ bool solve_5x5 (double A[5][5], double b[5])
   {
   // First Gaussian-Jordan elimination to put in echelon form
   // printf ("In solve_5x5: A\n");
   // printAb(A, b);

   // For debugging
   double Aorig[5][5];
   memcpy (&(Aorig[0][0]), A, sizeof(Aorig));
   double borig[5];
   memcpy (&(borig[0]), &(b[0]), sizeof(borig));

   // printf ("b\n");
   // printf ("%8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n",b[0],b[1],b[2],b[3],b[4]);

   const int nrows = 5, ncols = 5;
   for (int row=0; row<5; row++)
      {
      // Find the maximum element in the column
      const int col = row;
      double maxval = fabs(A[row][col]);
      int maxrow = row;
       
      // Find the maximum value in the column
      for (int i=row+1; i<nrows; i++)
         {
         double val = fabs(A[i][col]);
         if (val > maxval)
            {
            maxval = val;
            maxrow = i;
            }
         }

      // printf ("In solve_5x5: row %d\n", row);

      // Pivot
      if (row != maxrow)
         {
         // Swap rows row and maxrow
         double t;
         for (int j=col; j<ncols; j++)
            {
            t = A[row][j];
            A[row][j] = A[maxrow][j];
            A[maxrow][j] = t;
            }

         // Swap elements in the vector b
         t = b[row];
         b[row] = b[maxrow];
         b[maxrow] = t;
         }

      // printf ("In solve_5x5: after pivot\n");
      // printAb(A, b);

      // Eliminate
      for (int i=row+1; i<nrows; i++)
         {
         double fac = A[i][col] / A[row][col];
         for (int j=row+1; j<ncols; j++)
            A[i][j] -= fac * A[row][j];

         b[i] -= fac * b[row];
         }

      // printf ("In solve_5x5: after elim\n");
      // printAb(A, b);
      }

   // printf ("In solve_5x5: B - in row-echelon-form\n");
   // printAb(A, b);

   // Now it is in row-echelon form, so do back-factorization
   for (int i=nrows-1; i>=0; i--)
      {
      // Substitute previous values
      for (int j=i+1; j<ncols; j++)
         b[i] -= A[i][j] * b[j];

      // Update the vector b
      b[i] /= A[i][i];
      }

   // Check that all numbers are valid
   bool good = true;
   for (int i=0; i<5; i++)
      if (0) //! isfinite(b[i])) 
         {
         b[i] = 0.0;
         good = false;
         }

#ifdef RH_DEBUG
   // printf ("In solve_5x5: finished - Now test\n");
   for (int i=0; i<5; i++)
      {
      double val = 0.0;
      for (int j=0; j<5; j++)
         val += Aorig[i][j] * b[j];
      printf ("%13.4e  %13.4e %13.4e %13.4e\n", 
         val, borig[i], val - borig[i], b[i]);
      }
#endif

   // Return success or not
   return good;
   }

__host__ __device__ void update (Ematrix U, Ematrix V, double e[5])
   {
   // Updates U to the new value
   //   U -> U . Rz'. Ry' . Rx'
   //   V -> V . Ru' . Rv'
   // Order of params is x, y, z, v, u
   Gright(U, 0, 1, e[2]);  // z
   Gright(U, 0, 2, e[1]);  // y
   Gright(U, 1, 2, e[0]);  // x

   // V -> V . Rv' . Ru'
   Gright(V, 1, 2, e[4]);  // u
   Gright(V, 0, 2, e[3]);  // v
   }

__host__ __device__ double sq_error (Ematrix E, Matches_5 pin, Matches_5 qin)
   {
   // Computes the error for the points
   double errsq = 0.0;
   for (int pt=0; pt<5; pt++)
      {
      // Compute (q . E . p)^2
      double err = 0.0;
      for (int i=0; i<3; i++)
         for (int j=0; j<3; j++)
            err += E[i][j]*qin[pt][i]*pin[pt][j];
      errsq += err*err;
      }
   return errsq;
   }

template<int n>
__host__ __device__ double sq_error (Ematrix E, Matches_n<n> pin, Matches_n<n> qin)
   {
   // Computes the error for the points
   double errsq = 0.0;
   for (int pt=0; pt<n; pt++)
      {
      // Compute (q . E . p)^2
      double err = 0.0;
      for (int i=0; i<3; i++)
         for (int j=0; j<3; j++)
            err += E[i][j]*qin[pt][i]*pin[pt][j];
      errsq += err*err;
      }
   return errsq;
   }

__host__ __device__ void polish_E (Ematrix E, Matches_5 pin, Matches_5 qin, const int MaxReps)
   {
   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   // printf ("In polish_E\n");
   // printf ("U and V\n");
   // printE(U);
   // printE(V);

   // Go into a loop of polishing
   double oldemag = 1.0e6; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      Matches_5 p, q;
      for (int i=0; i<5; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[i][0]*V[0][j] + pin[i][1]*V[1][j] + pin[i][2]*V[2][j];
            q[i][j] = qin[i][0]*U[0][j] + qin[i][1]*U[1][j] + qin[i][2]*U[2][j];
            }

      // Print out the points
      // printf ("Transformed p and q\n");
      // printM(p);
      // printM(q);

      // Form the vector
      double err[5], emag = 0.0;
      for (int i=0; i<5; i++)
         {
         err[i] = -(p[i][0]*q[i][0] + p[i][1]*q[i][1]);
         emag += err[i]*err[i];
         }

      // Now, if emag is greater than old mag, then we stop
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      // printf ("E = \n");
      // printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double J[5][5];
      for (int j=0; j<5; j++) // j is the point number
         {
          J[j][0] = -p[j][1] * q[j][2];                    // -py*qz
          J[j][1] = -p[j][0] * q[j][2];                    // -px*qz
          J[j][2] =  p[j][1] * q[j][0] - p[j][0] * q[j][1];// py*qx - px*qy
          J[j][3] = -p[j][2] * q[j][0];                    // -pz*qx
          J[j][4] = -p[j][2] * q[j][1];                    // -pz*qy
         }

      // Solve the equation
      solve_5x5 (J, err);

      // printf ("del = \n");
      // for (int i=0; i<5; i++)
      //   printf ("%12.4e  ", err[i]);
      // printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, err);

      // printf ("U and V after update\n");
      // printE(U);
      // printE(V);
      }
   }

/*
 * Iterative least squares refinement of E matrix
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
template<int n>
__host__ __device__ void polish_E (Ematrix E, const double* pin, const double* qin, const int MaxReps)
   {
   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   printf ("In polish_E\n");
   printf ("U and V\n");
   printE(U);
   printE(V);

   // Go into a loop of polishing
   double oldemag = 1.0e8; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      Matches_n<n> p, q;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      // printf ("Transformed p and q\n");
      // printM(p);
      // printM(q);

      // Form the vector -J^T * err
      double JTerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTerr[i] = 0.0; // But just to be absolutely sure
      for (int k=0; k<n; k++) { // Loop over the points
        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        JTerr[0] += -p[k][1] * q[k][2] * -errk;
        JTerr[1] += -p[k][0] * q[k][2] * -errk;
        JTerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk;
        JTerr[3] += -p[k][2] * q[k][0] * -errk;
        JTerr[4] += -p[k][2] * q[k][1] * -errk;
      }


      printf("JTerr \n");
      for (int i = 0; i < 5; i++)
        printf("%f ", JTerr[i]);


      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTerr[i]*JTerr[i];
      emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      // printf("emag %f \n", emag);
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      // printf ("E = \n");
      // printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTJ[i][j] += Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTJ, JTerr);

      printf ("del = \n");
      for (int i=0; i<5; i++)
        printf ("%12.4e  ", JTerr[i]);
      printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTerr);

      printf ("U and V after update\n");
      printE(U);
      printE(V);
      }
   }

/*
 * Iteratively reweighted least squares refinement of E matrix
 * With Huber weighting for robustness J^T W J Delta = -J^T W epsilon
 * See: https://newonlinecourses.science.psu.edu/stat501/node/353/
 * delta: scale parameter - size of residual when behaviour changes from quadratic to linear
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
template<int n>
__host__ __device__ void polish_E_huber (Ematrix E, double* pin, double* qin, const double delta, const int MaxReps)
   {
   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   printf ("In polish_E\n");
   printf ("U and V\n");
   printE(U);
   printE(V);

   // Go into a loop of polishing
   double oldemag = 1.0e7; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      Matches_n<n> p, q;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      // printf ("Transformed p and q\n");
      // printM(p);
      // printM(q);

      // Form the vector -J^T * err
      double JTWerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTWerr[i] = 0.0; // But just to be absolutely sure
      double weights[n];
      for (int k=0; k<n; k++) { // Loop over the points
        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        weights[k] = (fabs(errk) < delta) ? 1.0 : delta / fabs(errk);
        JTWerr[0] += -p[k][1] * q[k][2] * -errk * weights[k];
        JTWerr[1] += -p[k][0] * q[k][2] * -errk * weights[k];
        JTWerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk * weights[k];
        JTWerr[3] += -p[k][2] * q[k][0] * -errk * weights[k];
        JTWerr[4] += -p[k][2] * q[k][1] * -errk * weights[k];
      }


      // printf("JTerr \n");
      // for (int i = 0; i < 5; i++)
      //   printf("%f ", JTWerr[i]);

      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTWerr[i]*JTWerr[i];
      emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      // printf("emag %f \n", emag);
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      // printf ("E = \n");
      // printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTWJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTWJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTWJ[i][j] += weights[k] * Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTWJ, JTWerr);

      // printf ("del = \n");
      // for (int i=0; i<5; i++)
      //   printf ("%12.4e  ", JTWerr[i]);
      // printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTWerr);

      // printf ("U and V after update\n");
      // printE(U);
      // printE(V);
      }
   }

/*
 * Iteratively reweighted least squares refinement of E matrix
 * With Huber weighting for robustness J^T W J Delta = -J^T W epsilon
 * See: https://newonlinecourses.science.psu.edu/stat501/node/353/
 * Scales residuals by tau = MAD / 0.6745 (Median Absolute Deviation), so that a fixed scale parameter
 * delta can be used (1.345), for 95% efficiency assuming the inliers have Gaussian noise
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
template<int n>
__host__ __device__ void polish_E_huber (Ematrix E, double* pin, double* qin, const int MaxReps)
   {
   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   // printf ("In polish_E\n");
   // printf ("U and V\n");
   // printE(U);
   // printE(V);

   // Go into a loop of polishing
   double oldemag = 1.0e7; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      Matches_n<n> p, q;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      // printf ("Transformed p and q\n");
      // printM(p);
      // printM(q);

      // Compute error vector and scale parameter tau
      // tau = median(r_i - r_bar) / 0.6745
      // r_i = abs(err_i); r_bar = median({r_i})
      double err[n];
      double abs_err[n];
      for (int k=0; k<n; k++) { // Loop over the points
        err[k] = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        abs_err[k] = fabs(err[k]);
      }
      int middle_index = 0.5 * n;
//      std::nth_element(abs_err, abs_err + middle_index, abs_err + n); // Use this for host-only code
//      double median_abs_err = abs_err[middle_index]; // Use this for host-only code
      double median_abs_err = quickselect(abs_err, middle_index, n);
      for (int k=0; k<n; k++) { // Loop over the points
        abs_err[k] = fabs(abs_err[k] - median_abs_err);
      }
//      std::nth_element(abs_err, abs_err + middle_index, abs_err + n); // Use this for host-only code
//      double tau = abs_err[middle_index] / 0.6745; // Use this for host-only code
      double tau = quickselect(abs_err, middle_index, n) / 0.6745;
      double delta = 1.345;

      // printf("median_abs_err: %f, tau: %f\n", median_abs_err, tau);

      // Form the vector -J^T * err
      double JTWerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTWerr[i] = 0.0; // But just to be absolutely sure
      double weights[n];
      for (int k=0; k<n; k++) { // Loop over the points
//        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        double errk = err[k];
        weights[k] = (fabs(errk) < tau * delta) ? 1.0 : tau * delta / fabs(errk);
        JTWerr[0] += -p[k][1] * q[k][2] * -errk * weights[k];
        JTWerr[1] += -p[k][0] * q[k][2] * -errk * weights[k];
        JTWerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk * weights[k];
        JTWerr[3] += -p[k][2] * q[k][0] * -errk * weights[k];
        JTWerr[4] += -p[k][2] * q[k][1] * -errk * weights[k];
      }


      // printf("JTerr \n");
      // for (int i = 0; i < 5; i++)
      //   printf("%f ", JTWerr[i]);

      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTWerr[i]*JTWerr[i];
      emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      // printf("emag %f \n", emag);
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      // printf ("E = \n");
      // printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTWJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTWJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTWJ[i][j] += weights[k] * Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTWJ, JTWerr);

      // printf ("del = \n");
      // for (int i=0; i<5; i++)
      //   printf ("%12.4e  ", JTWerr[i]);
      // printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTWerr);

      // printf ("U and V after update\n");
      // printE(U);
      // printE(V);
      }
   }

/*
 * Iteratively reweighted least squares refinement of E matrix
 * With parametric robust weighting J^T W J Delta = -J^T W epsilon
 * alpha in [0, 1], smooth transition between truncated L2 and Huber penalty functions
 * alpha = 0 ==> truncated L2
 * alpha = 1 ==> Huber norm
 * See: https://newonlinecourses.science.psu.edu/stat501/node/353/
 * delta: scale parameter - size of residual when behaviour changes from quadratic to linear
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
template<int n>
__host__ __device__ void polish_E_robust_parametric (Ematrix E, const double* pin, const double* qin, const double delta, const double alpha, const int MaxReps)
   {
   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   printf ("In polish_E\n");
   printf ("U and V\n");
   printE(U);
   printE(V);

   // Go into a loop of polishing
   double oldemag = 1.0e7; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      Matches_n<n> p, q;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      printf ("Transformed p and q\n");
      printM(p);
      printM(q);

      // Form the vector -J^T * err
      double JTWerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTWerr[i] = 0.0; // But just to be absolutely sure
      double weights[n];
      for (int k=0; k<n; k++) { // Loop over the points
        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        weights[k] = (fabs(errk) < delta) ? 1.0 : alpha * delta / fabs(errk);
        JTWerr[0] += -p[k][1] * q[k][2] * -errk * weights[k];
        JTWerr[1] += -p[k][0] * q[k][2] * -errk * weights[k];
        JTWerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk * weights[k];
        JTWerr[3] += -p[k][2] * q[k][0] * -errk * weights[k];
        JTWerr[4] += -p[k][2] * q[k][1] * -errk * weights[k];
      }


      printf("JTerr \n");
      for (int i = 0; i < 5; i++)
        printf("%f ", JTWerr[i]);

      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTWerr[i]*JTWerr[i];
      emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      // printf("emag %f \n", emag);
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      printf ("E = \n");
      printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTWJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTWJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTWJ[i][j] += weights[k] * Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTWJ, JTWerr);

      printf ("del = \n");
      for (int i=0; i<5; i++)
        printf ("%12.4e  ", JTWerr[i]);
      printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTWerr);

      printf ("U and V after update\n");
      printE(U);
      printE(V);
      }
   }

/*
 * Iteratively reweighted least squares refinement of E matrix
 * With parametric robust weighting J^T W J Delta = -J^T W epsilon
 * alpha in (-inf, 2], smooth transition between Welsch loss and L2 loss
 * alpha = -inf ==> Welsch
 * alpha = -2 ==> Geman-McClure
 * alpha = 1 ==> pseudo-Huber
 * alpha = 2 ==> L2
 * See: https://newonlinecourses.science.psu.edu/stat501/node/353/
 * See: https://arxiv.org/pdf/1701.03077.pdf
 * delta: scale parameter - size of residual when behaviour changes from quadratic to linear
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
template<int n>
__host__ __device__ void polish_E_robust_parametric_barron (Ematrix E, const double* pin, const double* qin, const double delta, const double alpha, const int MaxReps)
   {
   double eps = 1e-5;
   double b = fabs(2.0 - alpha) + eps;
   double d = alpha >= 0.0 ? alpha + eps : alpha - eps;
   double delta2 = delta * delta;

   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   printf ("In polish_E\n");
   printf ("U and V\n");
   printE(U);
   printE(V);

   // Go into a loop of polishing
   double oldemag = 1.0e7; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      Matches_n<n> p, q;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      printf ("Transformed p and q\n");
      printM(p);
      printM(q);

      // Form the vector -J^T * err
      double JTWerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTWerr[i] = 0.0; // But just to be absolutely sure
      double weights[n];
      for (int k=0; k<n; k++) { // Loop over the points
        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        weights[k] = pow(errk * errk / delta2 / b + 1.0, 0.5 * d - 1.0) / delta2;
        JTWerr[0] += -p[k][1] * q[k][2] * -errk * weights[k];
        JTWerr[1] += -p[k][0] * q[k][2] * -errk * weights[k];
        JTWerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk * weights[k];
        JTWerr[3] += -p[k][2] * q[k][0] * -errk * weights[k];
        JTWerr[4] += -p[k][2] * q[k][1] * -errk * weights[k];
      }


      printf("JTerr \n");
      for (int i = 0; i < 5; i++)
        printf("%f ", JTWerr[i]);

      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTWerr[i]*JTWerr[i];
      emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      // printf("emag %f \n", emag);
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      printf ("E = \n");
      printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTWJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTWJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTWJ[i][j] += weights[k] * Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTWJ, JTWerr);

      printf ("del = \n");
      for (int i=0; i<5; i++)
        printf ("%12.4e  ", JTWerr[i]);
      printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTWerr);

      printf ("U and V after update\n");
      printE(U);
      printE(V);
      }
   }

/*
 * Basic non-recursive quickselect implementation
 * Modified from http://blog.teamleadnet.com/2012/07/quick-select-algorithm-find-kth-element.html
 */
__host__ __device__ double quickselect(double *array, int k, int n) {

  int from = 0;
  int to = n - 1;

 // if from == to we reached the kth element
 while (from < to) {
  int r = from;
  int w = to;
  double mid = array[(r + w) / 2];

  // stop if the reader and writer meets
  while (r < w) {
   if (array[r] >= mid) { // put the large values at the end
    double tmp = array[w];
    array[w] = array[r];
    array[r] = tmp;
    w--;
   } else { // the value is smaller than the pivot, skip
    r++;
   }
  }

  // if we stepped up (r++) we need to step one down
  if (array[r] > mid)
   r--;

  // the r pointer is on the end of the first k elements
  if (k <= r) {
   to = r;
  } else {
   from = r + 1;
  }
 }

 return array[k];
}

/*
 * Functions with Dynamic Memory Allocation
 */

/*
 * Iterative least squares refinement of E matrix
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
__host__ void polish_E (Ematrix E, const double* pin, const double* qin, const int n, const int MaxReps)
   {
   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   printf ("In polish_E\n");
   printf ("U and V\n");
   printE(U);
   printE(V);

//   std::vector<std::array<double, 3>> p(n);
//   std::vector<std::array<double, 3>> q(n);
   double (*p)[3] = new double[n][3];
   double (*q)[3] = new double[n][3];

   // Go into a loop of polishing
   double oldemag = 1.0e8; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      // printf ("Transformed p and q\n");
      // printM(p);
      // printM(q);

      // Form the vector -J^T * err
      double JTerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTerr[i] = 0.0; // But just to be absolutely sure
      for (int k=0; k<n; k++) { // Loop over the points
        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        JTerr[0] += -p[k][1] * q[k][2] * -errk;
        JTerr[1] += -p[k][0] * q[k][2] * -errk;
        JTerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk;
        JTerr[3] += -p[k][2] * q[k][0] * -errk;
        JTerr[4] += -p[k][2] * q[k][1] * -errk;
      }


      printf("JTerr \n");
      for (int i = 0; i < 5; i++)
        printf("%f ", JTerr[i]);


      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTerr[i]*JTerr[i];
      emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      // printf("emag %f \n", emag);
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      // printf ("E = \n");
      // printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTJ[i][j] += Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTJ, JTerr);

      printf ("del = \n");
      for (int i=0; i<5; i++)
        printf ("%12.4e  ", JTerr[i]);
      printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTerr);

      printf ("U and V after update\n");
      printE(U);
      printE(V);
      }

   // Release dynamically-allocated memory
   delete[] p;
   delete[] q;
   }

/*
 * Iteratively reweighted least squares refinement of E matrix
 * With Huber weighting for robustness J^T W J Delta = -J^T W epsilon
 * See: https://newonlinecourses.science.psu.edu/stat501/node/353/
 * delta: scale parameter - size of residual when behaviour changes from quadratic to linear
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
__host__ void polish_E_huber (Ematrix E, const double* pin, const double* qin, const int n, const double delta, const int MaxReps)
   {
   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   printf ("In polish_E\n");
   printf ("U and V\n");
   printE(U);
   printE(V);

//   std::vector<std::array<double, 3>> p(n);
//   std::vector<std::array<double, 3>> q(n);
//   std::vector<double> weights(n);
   double (*p)[3] = new double[n][3];
   double (*q)[3] = new double[n][3];
   double *weights = new double[n];

   // Go into a loop of polishing
   double oldemag = 1.0e7; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      printf ("Transformed p and q\n");
      printM(p);
      printM(q);

      // Form the vector -J^T * err
      double JTWerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTWerr[i] = 0.0; // But just to be absolutely sure
      for (int k=0; k<n; k++) { // Loop over the points
        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        weights[k] = (fabs(errk) < delta) ? 1.0 : delta / fabs(errk);
        JTWerr[0] += -p[k][1] * q[k][2] * -errk * weights[k];
        JTWerr[1] += -p[k][0] * q[k][2] * -errk * weights[k];
        JTWerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk * weights[k];
        JTWerr[3] += -p[k][2] * q[k][0] * -errk * weights[k];
        JTWerr[4] += -p[k][2] * q[k][1] * -errk * weights[k];
      }


      printf("JTerr \n");
      for (int i = 0; i < 5; i++)
        printf("%f ", JTWerr[i]);

      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTWerr[i]*JTWerr[i];
      emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      // printf("emag %f \n", emag);
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      printf ("E = \n");
      printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTWJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTWJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTWJ[i][j] += weights[k] * Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTWJ, JTWerr);

      printf ("del = \n");
      for (int i=0; i<5; i++)
        printf ("%12.4e  ", JTWerr[i]);
      printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTWerr);

      printf ("U and V after update\n");
      printE(U);
      printE(V);
      }

   // Release dynamically-allocated memory
   delete[] p;
   delete[] q;
   delete[] weights;
   }

/*
 * Iteratively reweighted least squares refinement of E matrix
 * With parametric robust weighting J^T W J Delta = -J^T W epsilon
 * alpha in [0, 1], smooth transition between truncated L2 and Huber penalty functions
 * alpha = 0 ==> truncated L2
 * alpha = 1 ==> Huber norm
 * See: https://newonlinecourses.science.psu.edu/stat501/node/353/
 * delta: scale parameter - size of residual when behaviour changes from quadratic to linear
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
__host__ void polish_E_robust_parametric (Ematrix E, const double* pin, const double* qin, const int n, const double delta, const double alpha, const int MaxReps)
   {
   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   // printf ("In polish_E\n");
   // printf ("U and V\n");
   // printE(U);
   // printE(V);

//   std::vector<std::array<double, 3>> p(n);
//   std::vector<std::array<double, 3>> q(n);
//   std::vector<double> weights(n);
   double (*p)[3] = new double[n][3];
   double (*q)[3] = new double[n][3];
   double *weights = new double[n];

   // Go into a loop of polishing
   double oldemag = 1.0e20; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      // printf ("Transformed p and q\n");
      // printM(p);
      // printM(q);

      // Form the vector -J^T * err
      double JTWerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTWerr[i] = 0.0; // But just to be absolutely sure
      for (int k=0; k<n; k++) { // Loop over the points
        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        weights[k] = (fabs(errk) < delta) ? 1.0 : alpha * delta / fabs(errk);
        JTWerr[0] += -p[k][1] * q[k][2] * -errk * weights[k];
        JTWerr[1] += -p[k][0] * q[k][2] * -errk * weights[k];
        JTWerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk * weights[k];
        JTWerr[3] += -p[k][2] * q[k][0] * -errk * weights[k];
        JTWerr[4] += -p[k][2] * q[k][1] * -errk * weights[k];
      }

      // printf("JTerr \n");
      // for (int i = 0; i < 5; i++)
      //   printf("%f ", JTWerr[i]);
      // printf("\n");

      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTWerr[i]*JTWerr[i];
      // emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      printf("emag %.24f \n", emag);
      // printf("old emag %.30f \n", oldemag);
      // if (emag > oldemag || emag < 1.0e-24) break;
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      // printf ("E = \n");
      // printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTWJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTWJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTWJ[i][j] += weights[k] * Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTWJ, JTWerr);

      // printf ("del = \n");
      // for (int i=0; i<5; i++)
      //   printf ("%12.4e  ", JTWerr[i]);
      // printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTWerr);

      // printf ("U and V after update\n");
      // printE(U);
      // printE(V);
      }

   // Release dynamically-allocated memory
   delete[] p;
   delete[] q;
   delete[] weights;
   }

/*
 * Iteratively reweighted least squares refinement of E matrix
 * With parametric robust weighting J^T W J Delta = -J^T W epsilon
 * alpha in (-inf, 2], smooth transition between Welsch loss and L2 loss
 * alpha = -inf ==> Welsch
 * alpha = -2 ==> Geman-McClure
 * alpha = 1 ==> pseudo-Huber
 * alpha = 2 ==> L2
 * See: https://newonlinecourses.science.psu.edu/stat501/node/353/
 * See: https://arxiv.org/pdf/1701.03077.pdf
 * delta: scale parameter - size of residual when behaviour changes from quadratic to linear
 * Note that the Jacobian for each point is slightly different to that in Eq (13) of the PAMI'12 paper,
 * up to some sign changes and swaps. I'm taking the values in the above function as gospel, for now.
 * Presumably because the decomposition is defined slightly differently.
 */
__host__ void polish_E_robust_parametric_barron (Ematrix E, const double* pin, const double* qin, const int n, const double delta, const double alpha, const int MaxReps)
   {
   double eps = 1e-5;
   double b = fabs(2.0 - alpha) + eps;
   double d = alpha >= 0.0 ? alpha + eps : alpha - eps;
   double delta2 = delta * delta;

   // Decompose the matrix
   Ematrix U, V;
   Edecomp (E, U, V);

   printf ("In polish_E\n");
   printf ("U and V\n");
   printE(U);
   printE(V);

//   std::vector<std::array<double, 3>> p(n);
//   std::vector<std::array<double, 3>> q(n);
//   std::vector<double> weights(n);
   double (*p)[3] = new double[n][3];
   double (*q)[3] = new double[n][3];
   double *weights = new double[n];

   // Go into a loop of polishing
   double oldemag = 1.0e7; // Large value
   for (int rep=0; ; rep++)
      {
      // Multiply out the matches: q <- q . U; p <- p . V;
      for (int i=0; i<n; i++)
         for (int j=0; j<3; j++)
            {
            p[i][j] = pin[2*i+0]*V[0][j] + pin[2*i+1]*V[1][j] + 1.0*V[2][j];
            q[i][j] = qin[2*i+0]*U[0][j] + qin[2*i+1]*U[1][j] + 1.0*U[2][j];
            }

      // Print out the points
      printf ("Transformed p and q\n");
      printM(p);
      printM(q);

      // Form the vector -J^T * err
      double JTWerr[5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) JTWerr[i] = 0.0; // But just to be absolutely sure
      for (int k=0; k<n; k++) { // Loop over the points
        double errk = p[k][0]*q[k][0] + p[k][1]*q[k][1]; // epsilon
        weights[k] = pow(errk * errk / delta2 / b + 1.0, 0.5 * d - 1.0) / delta2;
        JTWerr[0] += -p[k][1] * q[k][2] * -errk * weights[k];
        JTWerr[1] += -p[k][0] * q[k][2] * -errk * weights[k];
        JTWerr[2] +=  (p[k][1] * q[k][0] - p[k][0] * q[k][1]) * -errk * weights[k];
        JTWerr[3] += -p[k][2] * q[k][0] * -errk * weights[k];
        JTWerr[4] += -p[k][2] * q[k][1] * -errk * weights[k];
      }


      printf("JTerr \n");
      for (int i = 0; i < 5; i++)
        printf("%f ", JTWerr[i]);

      double emag = 0.0;
      for (int i=0; i<5; i++) emag += JTWerr[i]*JTWerr[i];
      emag /= n; // divide error by n, to make it smaller
      // Now, if emag is greater than old mag, then we stop
      printf("emag %f \n", emag);
      if (emag > oldemag) break;
      oldemag = emag;


      // Compute the E from the present values of U, V
      Eprod (U, V, E);

      printf ("E = \n");
      printE(E);

      // Here is the break point, after it has really tried MaxReps changes
      if (rep == MaxReps) break;

      // Now, form a matrix
      double JTWJ[5][5] = {}; // Sets all elements to 0
      for (int i=0; i<5; i++) for (int j=0; j<5; j++) JTWJ[i][j] = 0.0; // But just to be absolutely sure

      for (int k=0; k<n; k++) { // Loop over the points
        double Jk[5];
        Jk[0] = -p[k][1] * q[k][2];                    // -py*qz
        Jk[1] = -p[k][0] * q[k][2];                    // -px*qz
        Jk[2] =  p[k][1] * q[k][0] - p[k][0] * q[k][1];// py*qx - px*qy
        Jk[3] = -p[k][2] * q[k][0];                    // -pz*qx
        Jk[4] = -p[k][2] * q[k][1];                    // -pz*qy
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            JTWJ[i][j] += weights[k] * Jk[i] * Jk[j];
          }
        }
      }

      // Solve the equation
      solve_5x5 (JTWJ, JTWerr);

      printf ("del = \n");
      for (int i=0; i<5; i++)
        printf ("%12.4e  ", JTWerr[i]);
      printf ("\n");

      // Now, build the new U and V from the params
      update (U, V, JTWerr);

      printf ("U and V after update\n");
      printE(U);
      printE(V);
      }

   // Release dynamically-allocated memory
   delete[] p;
   delete[] q;
   delete[] weights;
   }
