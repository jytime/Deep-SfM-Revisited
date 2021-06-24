#include <iostream>
#include <math.h>
#include <float.h>
#include <iterator>
#include <random>
#include <algorithm>

#include "common.h"
#include "polish_E.cu"
//#include "polydet.cu"
//#include "sturm.cu"
//#include "polyquotient.cu"
//#include "cheirality.cu"
#include "kernel_functions.cu"
// all CUDA definitions must be compiled in the same logical compilation unit
// Never compile FooDevice.cu - import it into main.cu

//#include <Utilities_CC/Timer.h>
#include <chrono>
#include <vector>
//#include "Utilities_CC/utilities_CC.h"



/*
 * CUDA macros, constants and functions
 */
#define CudaErrorCheck(ans) {__CudaErrorCheck((ans), __FILE__, __LINE__);}

void __CudaErrorCheck(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    std::cout << "CUDA Error (" << file << ":" << line << "): " 
	      << cudaGetErrorString(code) << std::endl;
    exit(code);
  }
}

// (28) Multiprocessors, (128) CUDA Cores/MP
// Maximum number of threads per multiprocessor:  2048
// Maximum number of threads per block:           1024
// Maximum number of threads: 57344

const int nBlocks = 256;
const int nThreadsPerBlock = 256;
const int nProcesses = nBlocks * nThreadsPerBlock;


// Two absolute value functions, in case not defined
//inline int rhAbs(int a) {return (a > 0 ? a : -a);}
//inline double rhAbs(double a) {return (a > 0.0 ? a : -a);}

// Function Declarations:
void GenerateMatches(const int nPoints, double (*qs)[2], double (*qps)[2]);
void GenerateMatches(const int nPoints, double (*Xs)[3], double R[3][3], 
                     double t[3], double (*qs)[2], double (*qps)[2]);
void PolluteData(double (*qs)[2], double (*qps)[2], const int nPoints,
                 const double noise_std, const double outlier_frac);
void ComputeErrorP(Pmatrix P, double R_gt[3][3], double t_gt[3], 
                   double &rotation_error, double &translation_error);
void ComputeErrorE(Ematrix E1, Ematrix E2, double &error);
void ConvertRt2E(double R[3][3], double t[3], Ematrix E);



int main(int argc, char *argv[]) {
  const int subset_size = 5; // Number of points in each subset of the data (>= 5)
                            // other points used for cheirality test only
  const int nPoints = 78600; // Number of total point correspondences
  const int num_test_points = 10;
  const int ransac_points = 1000; // Number of points used for RANSAC
  const int ransac_iter = 1;
  const double ransac_threshold = 0.05;

  // Generate two sets of points, all matching, to test
  double (*qs)[2] = new double[nPoints][2];
  double (*qps)[2] = new double[nPoints][2];
  double (*Xs)[3] = new double[nPoints][3];
  // double (*Xs)[3];
  double R_gt[3][3];
  double t_gt[3];

  // GenerateMatches(nPoints, qs, qps);
  GenerateMatches(nPoints, Xs, R_gt, t_gt, qs, qps);

  double noise_std = 0.01;
  double outlier_frac = 0.1;
  PolluteData(qs, qps, nPoints, noise_std, outlier_frac);

  // CUDA Setup
  // Set GPU to use
  // int device = 0;
  // CudaErrorCheck(cudaSetDevice(device));

  cudaSetDevice(qs.get_device());

  int *num_inliers;
  double *qs_ptr;
  double *qps_ptr;
  double (*essential_matrices)[3][3];
  curandState* state;

  CudaErrorCheck(cudaMallocManaged((void **) &qs_ptr, 2 * nPoints * sizeof(double)));
  CudaErrorCheck(cudaMallocManaged((void **) &qps_ptr, 2 * nPoints * sizeof(double)));
  CudaErrorCheck(cudaMallocManaged((void **) &state, nProcesses * sizeof(curandState)));
  CudaErrorCheck(cudaMallocManaged((void **) &num_inliers, nProcesses * sizeof(int)));
  CudaErrorCheck(cudaMallocManaged((void **) &essential_matrices, nProcesses * 3 * 3 * sizeof(double)));

  for (int i = 0; i < nPoints; ++i) {
    for (int j = 0; j < 2; ++j) {
      qs_ptr[2 * i + j] = qs[i][j];
      qps_ptr[2 * i + j] = qps[i][j];
    }
  }

  // Copy constants to device constant memory
  CudaErrorCheck(cudaMemcpyToSymbol(c_num_points, &nPoints, sizeof(int)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_num_test_points, &num_test_points, sizeof(int)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_ransac_num_test_points, &ransac_points, sizeof(int)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_ransac_num_iterations, &ransac_iter, sizeof(int)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_inlier_threshold, &ransac_threshold, sizeof(double)));

  // Declare device versions of curandState and generate states, one for each process
  unsigned long long seed = 1234;
  SetupRandomState<<<nBlocks, nThreadsPerBlock>>>(seed, state);

  EstimateEssentialMatrix<subset_size><<<nBlocks, nThreadsPerBlock>>>(
                                                 qs_ptr,      // Two sets of matching points
                                                 qps_ptr,     // (flattened 2D arrays)
                                                 state,       // Random number generator state
                                                 num_inliers, // Number of inliers per thread
                                                 essential_matrices); // Essential matrices per thread

  CudaErrorCheck(cudaPeekAtLastError()); // Check for kernel launch error
  CudaErrorCheck(cudaDeviceSynchronize()); // Check for kernel execution error

  // Print number of E and P matrices
//  for (int i = 0; i < nProcesses; i++) { 
//    printf("Number of Ematrices: %d\n", nEMatrices[i]);
//  }


  /***** Testing Polish_E *****/

  // auto clock_begin = std::chrono::steady_clock::now();

  int ind_max = distance(num_inliers, max_element(num_inliers, num_inliers + nProcesses));
  cout << "The largest element is " << ind_max << '\n';
  cout << "The number of inliers: " << num_inliers[ind_max] << '\n';
  Ematrix E_best;
  memcpy(E_best, &essential_matrices[ind_max], sizeof(E_best));

  Ematrix E_before;
  memcpy(E_before, E_best, sizeof(E_before));

//  const int MaxReps = 10;
  //Matches_n<500> q_test, qp_test;
  //memcpy(q_test, qs, 3 * 500 * sizeof(double));
  //memcpy(qp_test, qps, 3 * 500 * sizeof(double));
  //polish_E<50000>(E_best, qs, qps, MaxReps);
//  polish_E<50000>(E_best, qs, qps, MaxReps);

  // auto duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_begin).count();
  // printf("Duration: %0.9fs \n", duration);

  Ematrix E_gt;

  ConvertRt2E(R_gt, t_gt, E_gt);

  printf("Before optimization\n");
  for (int j = 0; j < 3; j++) {
    printf("%f %f %f \n", E_before[j][0], E_before[j][1], E_before[j][2]);
  }
  printf("\n");

  printf("After optimization\n");
  for (int j = 0; j < 3; j++) {
    printf("%f %f %f \n", E_best[j][0], E_best[j][1], E_best[j][2]);
  }
  printf("\n");

  printf("Ground truth E\n");
  for (int j = 0; j < 3; j++) {
    printf("%f %f %f \n", E_gt[j][0], E_gt[j][1], E_gt[j][2]);
  }
  printf("\n");

  double error1;
  ComputeErrorE(E_gt, E_before, error1);
  double error2;
  ComputeErrorE(E_gt, E_best, error2);
  printf("Error between E before optimisation and GT: %f \n", error1);
  printf("Error between E after optimisation and GT: %f \n", error2);

  // Free Host Memory
  free(qs);
  free(qps);
  free(Xs);
  // Free Device Memory
  CudaErrorCheck(cudaFree(qs_ptr));
  CudaErrorCheck(cudaFree(qps_ptr));
  CudaErrorCheck(cudaFree(essential_matrices));
  CudaErrorCheck(cudaFree(num_inliers));
  CudaErrorCheck(cudaFree(state));
  
  return 0;
}

void ConvertRt2E(double R[3][3], double t[3], Ematrix E) {
  // E =[t]xR

  double t_skew[3][3] = {{0,    -t[2],   t[1]},
                         {t[2],     0,  -t[0]},
                         {-t[1], t[0],     0}};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double sum = 0;
      for (int k = 0; k < 3; k++) {
        sum += R[i][k] * t_skew[k][j]; 
      }
      E[i][j] = sum;
    }
  }
}


void ComputeErrorE(Ematrix E1, Ematrix E2, double &error) {
  // Compute Frobenius norm between two E matrices
  error = 0.0;
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
    error += (E1[i][j] / E1[2][2] - E2[i][j] / E2[2][2]) * (E1[i][j] / E1[2][2] - E2[i][j] / E2[2][2]);
  error = sqrt(error);
}

void ComputeErrorP(Pmatrix P, double R_gt[3][3], double t_gt[3], double &rotation_error, double &translation_error) {
  // Compute translation error (up to scale)
  double t[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
    t[i] += -P[j][i] * P[j][3]; // t = -R' * P4
    //t[i] = P[j][3];
  for (int i = 0; i < 3; ++i)
    translation_error += (t[i] / t[2] - t_gt[i] / t_gt[2]) * (t[i] / t[2] - t_gt[i] / t_gt[2]); // Normalize and compute squared error
  translation_error = sqrt(translation_error);

  // Compute rotation error
  // angle = arccos((trace(R'*R_gt) - 1) / 2)
  double cos_angle = 0.0;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
    cos_angle += P[i][j] * R_gt[i][j];
  cos_angle = (cos_angle - 1.0) / 2.0;
  rotation_error = (cos_angle >= 1.0) ? 0.0 : acos(cos_angle) * 180.0 / M_PI; // in degrees
}


void GenerateMatches(const int nPoints, double (*qs)[2], double (*qps)[2]) {
  for (int i = 0; i < nPoints; i++) {
    qs[i][0] = urandom();
    qs[i][1] = urandom();
  }

  for (int i = 0; i < nPoints; i++) {
    qps[i][0] = urandom();
    qps[i][1] = urandom();
  }
}

/*
 * Generate matched points.
 * First generate points in 3D. Then project points to two reference cameras. 
 * Points are in front of two cameras. 
 */
void GenerateMatches(const int nPoints, double (*Xs)[3], double R[3][3], double t[3], double (*qs)[2], double (*qps)[2]) {
  // 1. Generate random rotation and translation: R, t
  // Rotation matrix
  // First fill matrix with random entries
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R[i][j] = urandom();
    }
  }

  // Do QR factorization, using Gram-Schmidt. Q is rotation matrix 
  for (int row = 0; row < 3; row++) {
    // Normalize the row
    double sum = 0.0;
    for (int j = 0; j < 3; j++) sum += R[row][j] * R[row][j]; 
    double fac = 1.0 / sqrt(sum);
    for (int j = 0; j < 3; j++) R[row][j] *= fac;
    
    // Use to sweep out the subsequent rows
    for (int i = row + 1; i < 3; i++) {
      // Inner product of row i and row j
      double prod = 0.0;
      for (int j = 0; j < 3; j++)
        prod += R[row][j] * R[i][j]; // Inner product 
      for (int j = 0; j < 3; j++) R[i][j] -= prod * R[row][j];
    }
  }

  printf("Print R and t\n");
  printf("Rotation\n");
  for (int i = 0; i < 3; i++) {
    printf("%f %f %f\n", R[i][0], R[i][1], R[i][2]);
  }

//   //code verifying rotation matrix, R^{T} = R^{-1}
//   for (int i = 0; i < 3; i++) {
//     for (int j = 0; j < 3; j++) {
//       double row_sum = 0.0;
//       for (int k = 0; k < 3; k++) {
//         row_sum += R[i][k] * R[j][k];
//       }
//       printf ("%f\n", row_sum);
//     }
//   }
//  // Verify determinant == 1
//  double determinant = R[0][0] * R[1][1] * R[2][2]
//                     + R[0][1] * R[1][2] * R[2][0]
//                     + R[0][2] * R[1][0] * R[2][1]
//                     - R[0][2] * R[1][1] * R[2][0]
//                     - R[0][1] * R[1][0] * R[2][2]
//                     - R[0][0] * R[1][2] * R[2][1];
//  printf ("determinant: %f\n", determinant);
  
  // translation vector
  for (int i = 0; i < 3; i++) {
    t[i] = urandom();
//    t[i] = 0.0;
  }
  printf("Translation: %f %f %f\n", t[0], t[1], t[2]);

  // 2. Generate random 3D points in [-1,1]^3: Xs
  // 3. Transform 3D points Xps = R*(Xs - t)
  // 4. Force all points to be in front of both cameras
  double PXs[nPoints][3];
  for (int i = 0; i < nPoints; i++) {
    double depth = -1.0;
    while (depth <= 0) { // Loop until both points are in front of the camera
      Xs[i][0] = urandom();
      Xs[i][1] = urandom();
      Xs[i][2] = urandom() + 1.05; // Force points to be in front of the first camera

      // Transform 3D points Xps = R*(Xs - t)
      // Xs - t
      double X_sub_t[3];
      for (int j = 0; j < 3; j++) {
        X_sub_t[j] = Xs[i][j] - t[j];
      }
      // R*(Xs - t)
      for (int j = 0; j < 3; j++) {
        double sum = 0.0;
        for (int k = 0; k < 3; k++) {
          sum += R[j][k] * X_sub_t[k];
        }
        PXs[i][j] = sum;
      }
      
      // Transform 3D points Xps = R*Xs + t
      // for (int j = 0; j < 3; j++) {
      //   double sum = 0.0;
      //   for (int k = 0; k < 3; k++) {
      //     sum += R[j][k] * Xs[i][k];
      //   }
      //   PXs[i][j] = sum + t[j];
      // }
      depth = PXs[i][2];
    }
  }

  // 5. Project 3D points to 2D: qs, qps
  for (int i = 0; i < nPoints; i++) {
    qs[i][0] = Xs[i][0] / Xs[i][2];
    qs[i][1] = Xs[i][1] / Xs[i][2];

    qps[i][0] = PXs[i][0] / PXs[i][2];
    qps[i][1] = PXs[i][1] / PXs[i][2];
  }
}


void PolluteData(double (*qs)[2], double (*qps)[2], const int nPoints, const double noise_std, const double outlier_frac) {

  // Add noise to Data
  std::default_random_engine generator;
  std::normal_distribution<double> dist(0.0, noise_std);

  for (int i = 0; i < nPoints; i++) {
    qps[i][0] = qps[i][0] + dist(generator);
    qps[i][1] = qps[i][1] + dist(generator);
  } 

  // Create outliers in data
  int outlier_begin = (int) nPoints * (1 - outlier_frac);

  std::random_shuffle(&qps[outlier_begin], &qps[nPoints]);
}
