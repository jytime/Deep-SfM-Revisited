#include "./common.h"
#include "./essential_matrix_5pt_dcl.h"

__host__ __device__ void compute_P_matrices (
      Matches q, Matches qp, 
      Ematrix E[], 
      double *focal,    // Focal lengths. If null, then focal lengths are 1
      Pmatrix P[], 
      int &nsolutions,
      int npoints /*- = 5 -*/
   )
   {
   // Takes the E matrices and tries to find which ones are cheirally correct
   // Count the number of good solutions
   // rhTimer tim("find_P_matrices");

   int numP = 0;

   // Run through each of the Essential Matrices.
   for (int m=0; m<nsolutions; m++)
      {
      // Get the focal length, if given
      double finv = 1.0;
      if (focal) finv = 1.0 / focal[m];

      // Decompose into U and V
      double Ut[3][3];
      memset (&(Ut[0][0]), 0, sizeof(Ut));
      Ut[0][0] = Ut[1][1] = Ut[2][2] = 1.0;

      // Copy E into V
      double Vt[3][3];
      memcpy(&(Vt[0][0]), E[m], sizeof(Vt));

      // Givens rotation to delete point [1, 0], then [2,0]
      for (int i=0; i<=1; i++)      // Pivot on point (i, i)
         for (int k=i+1; k<3; k++)
            {
            double a = Vt[i][i];
            double b = Vt[k][i];
            double s = sqrt(a*a + b*b);

            // If the values are too small, then skip it
            if (s == 0.0) continue;

            // Scale the entries to length s
            a /= s;
            b /= s;

            // Now Givens rotate
            Vt[i][i] = s;
            Vt[k][i] = 0.0;

            // Modify other columns in rows i and k
            for (int j=i+1; j<3; j++)
               {
               double c = Vt[i][j];
               double d = Vt[k][j];
               Vt[i][j] = a*c + b*d;
               Vt[k][j] = a*d - b*c;
               }

            // We must also accumulate the matrix Ut
            if (k == 1)
               {
               // Set to the Givens matrix -- starting with the identity
               Ut[0][0] = Ut[1][1] = a;
               Ut[1][0] = -b; Ut[0][1] = b;
               Ut[2][2] = 1.0;
               }
            else if (k == 2)
               {
               for (int j=0; j<3; j++)
                  {
                  double temp = a*Ut[i][j] + b*Ut[k][j];
                  Ut[k][j]     =-b*Ut[i][j] + a*Ut[k][j];
                  Ut[i][j] = temp;
                  }
               }
            }

      // Normalize the rows of Vt to 1
      double scale = 1.0 / 
          sqrt(Vt[0][0]*Vt[0][0] + Vt[0][1]*Vt[0][1] + Vt[0][2]*Vt[0][2]);
      for (int i=0; i<2; i++) for (int j=0; j<3; j++) Vt[i][j] *= scale;

      // At this point we have carried out the Givens rotation
      // Now, fill out the other row of Vt
      Vt[2][0] = Vt[0][1]*Vt[1][2] - Vt[0][2]*Vt[1][1];
      Vt[2][1] = Vt[0][2]*Vt[1][0] - Vt[0][0]*Vt[1][2];
      Vt[2][2] = Vt[0][0]*Vt[1][1] - Vt[0][1]*Vt[1][0];

      // Counts the number in front
      int count0a = 0, count0b = 0;
      int count1a = 0, count1b = 0;

      // So, now we can handle each of the points
      for (int pt=0; pt<npoints; pt++) 
         {
         // Get the point
         double *x1 = &(q [pt][0]);
         double *x2 = &(qp[pt][0]);

         // Multiply them by Ut and Vt
         double Vx[3], Ux[3];
         Vx[0] = finv*x1[0]*Vt[0][0] + finv*x1[1]*Vt[0][1] + x1[2]*Vt[0][2];
         // Vx[1] = finv*x1[0]*Vt[1][0] + finv*x1[1]*Vt[1][1] + x1[2]*Vt[1][2];
         Vx[2] = finv*x1[0]*Vt[2][0] + finv*x1[1]*Vt[2][1] + x1[2]*Vt[2][2];

         // Ux[0] = finv*x2[0]*Ut[0][0] + finv*x2[1]*Ut[0][1] + x2[2]*Ut[0][2];
         Ux[1] = finv*x2[0]*Ut[1][0] + finv*x2[1]*Ut[1][1] + x2[2]*Ut[1][2];
         Ux[2] = finv*x2[0]*Ut[2][0] + finv*x2[1]*Ut[2][1] + x2[2]*Ut[2][2];

         double denom1 =  Vx[0]*Ux[2] + Vx[2]*Ux[1];
         double denom2 = -Vx[0]*Ux[2] + Vx[2]*Ux[1];

         if (-Ux[1]/denom1 > 0.0) count0a++;
         if ( Vx[0]/denom1 > 0.0) count0b++;

         if (-Ux[1]/denom2 > 0.0) count1a++;
         if (-Vx[0]/denom2 > 0.0) count1b++;
         }

      // Counts in both images
      int count0 = count0a + count0b;
      int count1 = count1a + count1b;

      //-------------------------------------------

      // Now, check any of the counts is zero or all
      if (count0 == 2*npoints)
         {
         // Solution 0 is good
         for (int i=0; i<3; i++)
             for (int j=0; j<3; j++)
                P[numP][i][j] = Ut[0][i] * Vt[1][j] - Ut[1][i] * Vt[0][j] +
                                Ut[2][i] * Vt[2][j];

         for (int i=0; i<3; i++) P[numP][i][3] = Ut[2][i];

         // Copy the E matrix, if necessary
         if (m > numP) 
            {
            memcpy(E[numP], E[m], sizeof(Ematrix));
            if (focal) focal[numP] = focal[m];
            }

         numP++;
         }

      else if (count0 == 0)
         {
         // Solution 2 is good (opposite t)
         for (int i=0; i<3; i++)
             for (int j=0; j<3; j++)
                P[numP][i][j] = Ut[0][i] * Vt[1][j] - Ut[1][i] * Vt[0][j] +
                                Ut[2][i] * Vt[2][j];

         for (int i=0; i<3; i++) P[numP][i][3] = -Ut[2][i];

         // Copy the E matrix, if necessary
         if (m > numP) 
            {
            memcpy(E[numP], E[m], sizeof(Ematrix));
            if (focal) focal[numP] = focal[m];
            }

         numP++;
         }

      else if (count1 == 2*npoints)
         {
         // Solution 0 is good
         for (int i=0; i<3; i++)
             for (int j=0; j<3; j++)
                P[numP][i][j] = -Ut[0][i] * Vt[1][j] + Ut[1][i] * Vt[0][j] +
                                 Ut[2][i] * Vt[2][j];

         for (int i=0; i<3; i++) P[numP][i][3] = Ut[2][i];

         // Copy the E matrix, if necessary
         if (m > numP) 
            {
            memcpy(E[numP], E[m], sizeof(Ematrix));
            if (focal) focal[numP] = focal[m];
            }

         numP++;
         }

      else if (count1 == 0)
         {
         // Solution 2 is good (opposite t)
         for (int i=0; i<3; i++)
             for (int j=0; j<3; j++)
                P[numP][i][j] = -Ut[0][i] * Vt[1][j] + Ut[1][i] * Vt[0][j] +
                                 Ut[2][i] * Vt[2][j];

         for (int i=0; i<3; i++) P[numP][i][3] = -Ut[2][i];

         // Copy the E matrix, if necessary
         if (m > numP) 
            {
            memcpy(E[numP], E[m], sizeof(Ematrix));
            if (focal) focal[numP] = focal[m];
            }

         numP++;
         }
      }

   // Update the number of solutions
   nsolutions = numP;
   }

__host__ __device__ void compute_P_matrices_5pt (
      Matches q, Matches qp, 
      Pmatrix P[], 
      int &nsolutions,
      bool optimized
   )
   {
   // Computes and returns the P2 (assuming P1 = [I | 0])
   Ematrix E[10];
   double *focal = (double *) 0;
   compute_E_matrices (q, qp, E, nsolutions, optimized);
   compute_P_matrices (q, qp, E, focal, P, nsolutions, 5);
   }
