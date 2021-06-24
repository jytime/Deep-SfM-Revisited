/*
 * Kernel Functions
 */
#include <curand_kernel.h>
#include <math.h> // truncf
#include <stdio.h> 

#include "polydet.cu"
#include "sturm.cu"
#include "polyquotient.cu"
#include "cheirality.cu"
#include "essential_matrix_5pt.cu"
//#include "essential_matrix_6pt.cu"

// Declare constant memory (64KB maximum)
__constant__ int c_num_points;
__constant__ int c_num_test_points;
__constant__ int c_ransac_num_test_points;
__constant__ int c_ransac_num_iterations;
__constant__ double c_inlier_threshold;

// Function declarations
__device__ int RandomInt(curandState* state,
                         const int global_index,
                         const int min_int,
                         const int max_int);

template<int n>
__device__ void SelectSubset(const double *qs,
                             const double *qps,
                             curandState* state,
                             const int global_index,
                             Matches_n<n>& q,
                             Matches_n<n>& qp);

template<typename T>
__device__ void ComputeError(const T *q,
                             const T *qp,
                             const Ematrix &E,
                             T &error);

/*
 * Initialise a state for each thread
 */
__global__ void SetupRandomState(const unsigned long long seed, curandState* state) {
  int global_index = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(seed, global_index, 0, &state[global_index]);
}

/*
 * Estimate the essential matrix, using the 5-point algorithm and RANSAC
 */
template <int subset_size>
__global__ void EstimateEssentialMatrix(const double *qs, // Two sets of matching points
                                        const double *qps, // (flattened 2D arrays)
                                        curandState* state, // Random number generator state
                                        int *num_inliers, // Number of inliers per thread
                                        double (*essential_matrices)[3][3]) { // Essential matrices with greatest number of inliers

  int global_index = threadIdx.x + blockDim.x * blockIdx.x;

  int num_essential_matrices;
  Matches_n<subset_size> q, qp;
  Ematrix essential_matrix;
  Ematrix essential_matrix_set[10];

  // RANSAC
  int best_num_inliers = 0;
  for (int i = 0; i < c_ransac_num_iterations; ++i) {
    // Generate hypothesis set
    SelectSubset<subset_size>(qs, qps, state, global_index, q, qp);

    // Compute up to 10 essential matrices using the 5 point algorithm
    compute_E_matrices_optimized(q, qp, essential_matrix_set, num_essential_matrices);

    
    /**
    // Polish all essential matrices
    const double PolishThreshold = 5.0e-24;
    const int MaxReps = 7;        // Somewhat arbitrary bound
    for (int j = 0; j < num_essential_matrices; ++j) {
      double square_error = sq_error(essential_matrix_set[j], q, qp);
      if (square_error > PolishThreshold)
         polish_E(q, qp, essential_matrix_set[j], MaxReps);
    }

    // Find those that give correct cheirality
    int num_projection_matrices = num_essential_matrices;
    Pmatrix projection_matrix_set[10];
    compute_P_matrices(q, qp, essential_matrix_set, (double *)0, projection_matrix_set, num_projection_matrices, subset_size);
    **/

    // Test essential matrices in solution set
    // Choose the solution with the greatest number of inliers
    int best_num_inliers_subset = 0;
    int best_index = 0;
    for (int j = 0; j < num_essential_matrices; ++j) {
    // for (int j = 0; j < num_projection_matrices; ++j) {
      int inlier_count = 0;
      for (int k = 0; k < c_num_test_points; ++k) {
        double error;
        double q_test[3] = {qs[2 * k], qs[2 * k + 1], 1.0};
        double qp_test[3] = {qps[2 * k], qps[2 * k + 1], 1.0};
        ComputeError<double>(q_test, qp_test, essential_matrix_set[j], error);
        if (error <= c_inlier_threshold) {
          inlier_count++;
        }
      }
      if (inlier_count > best_num_inliers_subset) {
        best_num_inliers_subset = inlier_count;
        best_index = j;
      } 
    }

    // Evaluate best essential matrix on full set of test matches
    int inlier_count = 0;
    for (int k = 0; k < c_ransac_num_test_points; ++k) {
      double error;
      double q_test[3] = {qs[2 * k], qs[2 * k + 1], 1.0};
      double qp_test[3] = {qps[2 * k], qps[2 * k + 1], 1.0};
      ComputeError<double>(q_test, qp_test, essential_matrix_set[best_index], error);
      if (error <= c_inlier_threshold) {
        inlier_count++;
      }
    }
    if (inlier_count > best_num_inliers) {
      best_num_inliers = inlier_count;
      memcpy(essential_matrix, &essential_matrix_set[best_index], 3 * 3 * sizeof(double));
    }
  }
  
  // Copy to output
  num_inliers[global_index] = best_num_inliers;
  memcpy(&essential_matrices[global_index], essential_matrix, 3 * 3 * sizeof(double));
}

/*
 * Estimate the essential matrix and camera matrix, using the 5-point algorithm and RANSAC
 */
template <int subset_size>
__global__ void EstimateProjectionMatrix(const double *qs, // Two sets of matching points
                                        const double *qps, // (flattened 2D arrays)
                                        curandState* state, // Random number generator state
                                        int *num_inliers, // Number of inliers per thread
                                        double (*essential_matrices)[3][3], // Essential matrices with greatest number of inliers
                                        double (*projection_matrices)[3][4]){ // camera matrices with greates number of inliers

  int global_index = threadIdx.x + blockDim.x * blockIdx.x;

  int num_essential_matrices;
  Matches_n<subset_size> q, qp;
  Ematrix essential_matrix;
  Pmatrix projection_matrix;
  Ematrix essential_matrix_set[10];

  // RANSAC
  int best_num_inliers = 0;
  for (int i = 0; i < c_ransac_num_iterations; ++i) {
    // Generate hyposthesis set
    SelectSubset<subset_size>(qs, qps, state, global_index, q, qp);

    // Compute up to 10 essential matrices using the 5 point algorithm
    compute_E_matrices_optimized(q, qp, essential_matrix_set, num_essential_matrices);

    
    /**
    // Polish all essential matrices
    const double PolishThreshold = 5.0e-24;
    const int MaxReps = 7;        // Somewhat arbitrary bound
    for (int j = 0; j < num_essential_matrices; ++j) {
      double square_error = sq_error(essential_matrix_set[j], q, qp);
      if (square_error > PolishThreshold)
         polish_E(q, qp, essential_matrix_set[j], MaxReps);
    }
    **/

    // Find those that give correct cheirality
    int num_projection_matrices = num_essential_matrices;
    Pmatrix projection_matrix_set[10];
    compute_P_matrices(q, qp, essential_matrix_set, (double *)0, projection_matrix_set, num_projection_matrices, subset_size);

    // Test essential matrices in solution set
    // Choose the solution with the greatest number of inliers
    int best_num_inliers_subset = 0;
    int best_index = 0;
    // for (int j = 0; j < num_essential_matrices; ++j) {
    for (int j = 0; j < num_projection_matrices; ++j) {
      int inlier_count = 0;
      for (int k = 0; k < c_num_test_points; ++k) {
        double error;
        double q_test[3] = {qs[2 * k], qs[2 * k + 1], 1.0};
        double qp_test[3] = {qps[2 * k], qps[2 * k + 1], 1.0};
        ComputeError<double>(q_test, qp_test, essential_matrix_set[j], error);
        if (error <= c_inlier_threshold) {
          inlier_count++;
        }
      }
      if (inlier_count > best_num_inliers_subset) {
        best_num_inliers_subset = inlier_count;
        best_index = j;
      } 
    }

    // Evaluate best essential matrix on full set of test matches
    int inlier_count = 0;
    for (int k = 0; k < c_ransac_num_test_points; ++k) {
      double error;
      double q_test[3] = {qs[2 * k], qs[2 * k + 1], 1.0};
      double qp_test[3] = {qps[2 * k], qps[2 * k + 1], 1.0};
      ComputeError<double>(q_test, qp_test, essential_matrix_set[best_index], error);
      if (error <= c_inlier_threshold) {
        inlier_count++;
      }
    }
    if (inlier_count > best_num_inliers) {
      best_num_inliers = inlier_count;
      memcpy(essential_matrix, &essential_matrix_set[best_index], 3 * 3 * sizeof(double));
      memcpy(projection_matrix, &projection_matrix_set[best_index], 3 * 4 * sizeof(double));
    }
  }
  
  // Copy to output
  num_inliers[global_index] = best_num_inliers;
  memcpy(&essential_matrices[global_index], essential_matrix, 3 * 3 * sizeof(double));
  memcpy(&projection_matrices[global_index], projection_matrix, 3 * 4 * sizeof(double));
}

/*
 * Compute Sampson distance given a pair of matched points and an essential matrix
 */
template<typename T>
__device__ void ComputeError(const T *q,
                             const T *qp,
                             const Ematrix &E,
                             T &error) {
  // Compute Ex
  T Ex[3];
  for (int k = 0; k < 3; k++) {
    T sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += E[k][l] * q[l];
    Ex[k] = sum;
    }
  }
  // Compute x^TE
  T xE[3];
  for (int k = 0; k < 3; k++) {
    T sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += qp[l] * E[l][k];
    xE[k] = sum;
    }
  }
  // Compute xEx
  T xEx = 0.0;
  for (int k = 0; k < 3; k++) {
    xEx += qp[k] * Ex[k];
  }
  // Compute Sampson error
  T d = sqrt(Ex[0]*Ex[0]+Ex[1]*Ex[1]+xE[0]*xE[0]+xE[1]*xE[1]);
  error = xEx / d;

  if (error < 0.0) error = -error;
}

/*
 * Generate an integer in the range [min_int, max_int]
 */
__device__ int RandomInt(curandState* state,
                         const int global_index,
                         const int min_int,
                         const int max_int) {
  // Generate a random float in (0,1)
  float random_float = curand_uniform(&state[global_index]);
  random_float *= (max_int - min_int + 0.999999f);
  random_float += min_int;
  return (int) truncf(random_float);
}

/*
 * Generate a random subset of qs and qps, each thread selects a different subset
 * Optimised for speed, no checking that elements are unique
 */
template<int n>
__device__ void SelectSubset(const double* qs,
                             const double* qps,
                             curandState* state,
                             const int global_index,
                             Matches_n<n>& q,
                             Matches_n<n>& qp) {
  for (int i = 0; i < n; ++i) {
    int index = RandomInt(state, global_index, 0, c_num_points - 1);
    for (int j = 0; j < 2; ++j) {
      q[i][j] = qs[2 * index + j];
      qp[i][j] = qps[2 * index + j];
    }
    q[i][2] = 1.0;
    qp[i][2] = 1.0;
  }
}

/*
 * Generate a random subset of qs and qps, each thread selects a different subset
 * Ensures all elements are unique
 */
/*
template<int n>
__device__ void SelectSubset(const double* qs,
                             const double* qps,
                             curandState* state,
                             const bool random,
                             const int global_index,
                             Matches_n<n>& q,
                             Matches_n<n>& qp) {
  // Generate index set
  int index;
  int index_set[n] = {0};
  for (int i = 0; i < n; ++i) {
    bool index_in_set = true;
    while (index_in_set) {
      if (random)
        index = RandomInt(state, global_index, 0, c_num_points - 1);
      else
        index = (global_index + i) % c_num_points; // Debug, use sequential points
      index_in_set = false;
      for (int j = 0; j < i; ++j) { // Only look at previous entries
        if (index_set[j] == index) {
          index_in_set = true;
        }
      }
      if (!index_in_set) {
        index_set[i] = index;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 2; ++j) {
      q[i][j] = qs[2 * index_set[i] + j];
      qp[i][j] = qps[2 * index_set[i] + j];
    }
    q[i][2] = 1.0;
    qp[i][2] = 1.0;
  }
}
*/
