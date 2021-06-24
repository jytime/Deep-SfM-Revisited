#ifndef POLISH_E_H
#define POLISH_E_H

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "common.h"

__host__ __device__ void polish_E (
      Ematrix E, Matches_5 pin, Matches_5 qin, const int MaxReps = 5);

template<int n>
__host__ __device__ void polish_E (Ematrix E, const double* pin, const double* qin, const int MaxReps = 5);

template<int n>
__host__ __device__ void polish_E_huber (Ematrix E, const double* pin, const double* qin, const double delta, const int MaxReps = 5);

template<int n>
__host__ __device__ void polish_E_huber (Ematrix E, const double* pin, const double* qin, const int MaxReps = 5);

template<int n>
__host__ __device__ void polish_E_robust_parametric (Ematrix E, const double* pin, const double* qin, const double delta, const double alpha, const int MaxReps = 5);

template<int n>
__host__ __device__ void polish_E_robust_parametric_barron (Ematrix E, const double* pin, const double* qin, const double delta, const double alpha = 1.0, const int MaxReps = 5);

__host__ __device__ double sq_error (
      Ematrix E, Matches_5 pin, Matches_5 qin);

template<int n>
__host__ __device__ double sq_error (Ematrix E, Matches_n<n> pin, Matches_n<n> qin);

__host__ __device__ double quickselect(double *array, int k, int n);

__host__ void polish_E (Ematrix E, const double (*pin)[3], const double (*qin)[3], const int n, const int MaxReps);
__host__ void polish_E_huber (Ematrix E, const double (*pin)[3], const double (*qin)[3], const int n, const double delta, const int MaxReps);

//template<typename T>
//__host__ void polish_E_robust_parametric (Ematrix E, const T *pin, const T *qin, const int n, const T delta, const T alpha, const int MaxReps);
__host__ void polish_E_robust_parametric (Ematrix E, const double *pin, const double *qin, const int n, const double delta, const double alpha, const int MaxReps);

__host__ void polish_E_robust_parametric_barron (Ematrix E, const double (*pin)[3], const double (*qin)[3], const int n, const double delta, const double alpha, const int MaxReps);

__host__ __device__ void Edecomp(Ematrix E, double parameters[5]);

#endif
