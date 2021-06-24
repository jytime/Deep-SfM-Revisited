#ifndef ESSENTIAL_MATRIX6_CLASSES_H
#define ESSENTIAL_MATRIX6_CLASSES_H

#include <vector>
using namespace std;
#include "math.h"
#include "common.h"

#ifdef NO_TARGETJR
void test_E_matrix (const double E[3][3]) {}
#else
void test_E_matrix (const double E[3][3]);
#endif

// #define RH_DEBUG

// Some forward declarations
class poly3_1;
class poly3_2;
class poly3_3;
class EmatrixSet_6pt;

const int NMatches = 6;    // Number of point matches
const int Nvar = 9-NMatches;  // Number of free dof for E

// Define some variables that can be used in equations, etc
// These are in alphabetical order, assuming the order z x y
const int z_ = 0;
const int x_ = 1;
const int y_ = 2;

const int zz_ = 0;
const int zx_ = 1;
const int zy_ = 2;
const int xx_ = 3;
const int xy_ = 4;
const int yy_ = 5;

const int zzz_ = 0;
const int zzx_ = 1;
const int zzy_ = 2;
const int zxx_ = 3;
const int zxy_ = 4;
const int zyy_ = 5;
const int xxx_ = 6;
const int xxy_ = 7;
const int xyy_ = 8;
const int yyy_ = 9;

class poly3_3
   {
   protected :

      double A_[10];

   public :

      // const int size_;
      __host__ __device__ int size_() const {return 10;}
      __host__ __device__ poly3_3(){}

      __host__ __device__ poly3_3 operator + (poly3_3);
      __host__ __device__ void operator += (poly3_3);
      __host__ __device__ poly3_3 operator - (poly3_3);
      __host__ __device__ poly3_3 operator * (double);
      __host__ __device__ double &operator [] (int i) {return A_[i]; }


      __host__ __device__ void print ()
         {
         printf ("%12.3e\n%12.3e %12.3e\n", A_[zzz_], A_[zzx_], A_[zzy_]);
         printf ("%12.3e %12.3e %12.3e\n",  A_[zxx_], A_[zxy_], A_[zyy_]);
         printf ("%12.3e %12.3e %12.3e %12.3e\n", 
              A_[xxx_], A_[xxy_], A_[xyy_], A_[yyy_]);
         }

      __host__ __device__ void clear() 
         { 
         for (int i=0; i<size_(); i++) 
            A_[i] = 0.0;
         }

   };

class poly3_2 
   {
   protected :

      double A_[6];

   public :

      // const int size_;
      __host__ __device__ int size_() const {return 6;}
      __host__ __device__ poly3_2() {}

      __host__ __device__ poly3_3 operator * (poly3_1);
      __host__ __device__ poly3_2 operator + (poly3_2);
      __host__ __device__ void operator += (poly3_2);
      __host__ __device__ poly3_2 operator - (poly3_2);
      __host__ __device__ double &operator [] (int i) { return A_[i]; }

      __host__ __device__ void clear() 
         { 
         for (int i=0; i<size_(); i++) 
            A_[i] = 0.0;
         }

      __host__ __device__ void print ()
         {
         printf ("%12.3e\n%12.3e %12.3e\n", A_[zz_], A_[zx_], A_[zy_]);
         printf ("%12.3e %12.3e %12.3e\n",  A_[xx_], A_[xy_], A_[yy_]);
         }
   };

class poly3_1
   {
   protected :

      double A_[Nvar];

   public :

      // const int size_ ;
      __host__ __device__ int size_() const {return 3;}

      // Constructors
      __host__ __device__ poly3_1() {};
      __host__ __device__ poly3_1 (double x, double y, double z) 
         { A_[0] = x; A_[1] = y; A_[2] = z; }
      __host__ __device__ ~poly3_1 () {};

      // Operators
      __host__ __device__ poly3_2 operator * (poly3_1);
      __host__ __device__ poly3_1 operator + (poly3_1);
      __host__ __device__ poly3_1 operator - (poly3_1);
      __host__ __device__ double &operator [] (int i) { return A_[i]; }

      __host__ __device__ void clear() 
         { 
         for (int i=0; i<size_(); i++) 
            A_[i] = 0.0;
         }

      __host__ __device__ void print ()
         {
         printf ("%12.3e\n%12.3e %12.3e\n", A_[z_], A_[x_], A_[y_]);
         }
   };

class EmatrixSet_6pt
   {
   protected :
      poly3_1 E_[3][3];

   public :

      __host__ __device__ EmatrixSet_6pt () {};
      __host__ __device__ ~EmatrixSet_6pt() {};

      __host__ __device__ poly3_1 &operator () (int i, int j) { return E_[i][j]; }

      __host__ __device__ void print ()
         {
         for (int i=0; i<Nvar; i++)
            {
            for (int j=0; j<3; j++)
               {
               for (int k=0; k<3; k++)
             printf ("%12.3e ", E_[j][k][i]);
               printf ("\n");
               }
            printf ("\n");
            }
         }
   };

#endif
