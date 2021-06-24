#ifndef ESSENTIAL_MATRIX_5_CLASSES_H
#define ESSENTIAL_MATRIX_5_CLASSES_H

//#include <vector>
using namespace std;
#include "math.h"
#include "common.h"

// Some forward declarations
class poly4_1;
class poly4_2;
class poly4_3;
class EmatrixSet_5pt;

__host__ __device__ void print_equation_set (EquationSet A, int maxdegree = 3);
void test_E_matrix ( const double E[3][3]);

class poly4_3
   {
   protected :

      double A_[4][4][4];

   public :

      __host__ __device__ poly4_3 operator + (poly4_3);
      __host__ __device__ void operator += (poly4_3);
      __host__ __device__ poly4_3 operator - (poly4_3);
      __host__ __device__ poly4_3 operator * (double);
      __host__ __device__ double &operator () (int i, int j, int k) {return A_[i][j][k]; }

      __host__ __device__ void print ()
         {
         for (int i=0; i<4; i++) 
            {
            for (int j=0; j<4; j++) 
               {
               for (int k=0; k<4; k++) 
             printf ("%12.3e ", A_[i][j][k]);
               printf ("\n");
               }
            printf ("\n");
            }
         }

      __host__ __device__ void clear()
         { 
         for (int i=0; i<4; i++) 
            for (int j=0; j<4; j++) 
               for (int k=0; k<4; k++) 
               A_[i][j][k] = 0.0;
         }
   };

class poly4_2 
   {
   protected :

      double A_[4][4];

   public :
      __host__ __device__ poly4_3 operator * (poly4_1);
      __host__ __device__ poly4_2 operator + (poly4_2);
      __host__ __device__ void operator += (poly4_2);
      __host__ __device__ poly4_2 operator - (poly4_2);
      __host__ __device__ double &operator () (int i, int j) { return A_[i][j]; }

      __host__ __device__ void clear()
         { 
         for (int i=0; i<4; i++) 
            for (int j=0; j<4; j++) 
               A_[i][j] = 0.0;
         }

      __host__ __device__ void print ()
         {
         for (int i=0; i<4; i++) 
            {
            for (int j=0; j<4; j++) 
          printf ("%12.3e ", A_[i][j]);
            printf ("\n");
            }
         }
   };

class poly4_1
   {
   protected :

      double A_[4];

   public :

      // Constructors
      __host__ __device__ poly4_1(){};
      __host__ __device__ poly4_1 (double w, double x, double y, double z)
         { A_[0] = w; A_[1] = x; A_[2] = y; A_[3] = z; }
      __host__ __device__ ~poly4_1 () {};

      // Operators
      __host__ __device__ poly4_2 operator * (poly4_1);
      __host__ __device__ poly4_1 operator + (poly4_1);
      __host__ __device__ poly4_1 operator - (poly4_1);
      __host__ __device__ double &operator () (int i) { return A_[i]; }

      __host__ __device__ void print ()
         {
         for (int i=0; i<4; i++) 
            printf ("%12.3e ", A_[i]);
         printf ("\n");
         }
   };

class EmatrixSet_5pt
   {
   protected :

      poly4_1 E_[3][3];

   public :

      __host__ __device__ EmatrixSet_5pt () {};
      __host__ __device__ ~EmatrixSet_5pt() {};

      __host__ __device__ poly4_1 &operator () (int i, int j) { return E_[i][j]; }

      __host__ __device__ void print ()
         {
         for (int i=0; i<4; i++)
            {
            for (int j=0; j<3; j++)
               {
               for (int k=0; k<3; k++)
             printf ("%12.3e ", E_[j][k](i));
               printf ("\n");
               }
            printf ("\n");
            }
         }
   };

#endif
