#include <vector>
#include <iostream>
#include <chrono>
#include <tuple>
#include <ATen/ATen.h>

#include "common.h"
#include "polish_E.cu"
#include "kernel_functions.cu"

/*
 * CUDA macros, constants and functions
 */
const int subset_size = 5;
const unsigned long long seed = 1234;

#define CudaErrorCheck(ans) {__CudaErrorCheck((ans), __FILE__, __LINE__);}
void __CudaErrorCheck(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    std::cout << "CUDA Error (" << file << ":" << line << "): " 
	      << cudaGetErrorString(code) << std::endl;
    exit(code);
  }
}

/*
 * Decompose essential matrix into angles
 */
at::Tensor EssentialMatrixDecompose(
        at::Tensor Emat) {
    double *E_ptr = Emat.data_ptr<double>();
    Ematrix E;
    double parameter[5];
    memcpy(E, E_ptr, 3 * 3 * sizeof(double));

    Edecomp(E, parameter);

    at::Tensor parameters = at::empty(5, Emat.options());
    double *parameter_ptr = parameters.data_ptr<double>();
    memcpy(parameter_ptr, parameter, 5 * sizeof(double));

    return parameters;
}

/*
 * Decompose essential matrix into UI^VT, doing SVD.  
 */
std::tuple<at::Tensor, at::Tensor> EssentialMatrixDecomposeUV(
        at::Tensor Emat) {
    double *E_ptr = Emat.data_ptr<double>();
    Ematrix E;
    Ematrix U;
    Ematrix V;
    memcpy(E, E_ptr, 3 * 3 * sizeof(double));

    Edecomp(E, U, V);

    at::Tensor Umat = at::empty(3 * 3, Emat.options());
    at::Tensor Vmat = at::empty(3 * 3, Emat.options());
    double *U_ptr = Umat.data_ptr<double>();
    memcpy(U_ptr, U, 3 * 3 * sizeof(double));
    double *V_ptr = Vmat.data_ptr<double>();
    memcpy(V_ptr, V, 3 * 3 * sizeof(double));
    Umat.resize_({3,3});
    Vmat.resize_({3,3});

    auto t = std::make_tuple(Umat, Vmat);

    return t;
}


/*
 * Five point algorithm cuda optimization using robust cost functions
 */
at::Tensor EssentialMatrixOptimise(
        at::Tensor input1, // input 1 has size nx2, type double
        at::Tensor input2,
        at::Tensor initial_essential_matrix,
        const double delta,
        const double alpha,
        const int max_iterations) {

    auto clock_begin = std::chrono::steady_clock::now();

    const int num_points = input1.size(0);

    // Input data pointers
    double *input1_ptr = input1.data_ptr<double>();
    double *input2_ptr = input2.data_ptr<double>();
    double *essential_matrix_ptr = initial_essential_matrix.data_ptr<double>();

    Ematrix E_in;
    memcpy(E_in, essential_matrix_ptr, 3 * 3 * sizeof(double));

    polish_E_robust_parametric(E_in, input1_ptr, input2_ptr, num_points, delta, alpha, max_iterations);

    at::Tensor E_out = at::empty(3 * 3, initial_essential_matrix.options());
    double* outptr = E_out.data_ptr<double>();
    memcpy(outptr, E_in, 3 * 3 * sizeof(double));
    E_out.resize_({3,3});

    // std::cout << "Runtime (Optimise): " << std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_begin).count() << "s" << std::endl;
    return E_out; //E_optimised
}

/*
 * Five point algorithm cuda initialization 
 */
at::Tensor EssentialMatrixInitialise(
        at::Tensor input1, // input 1 has size nx2, type double
        at::Tensor input2,
        const int num_test_points, // 10
        const int num_ransac_test_points, // 1000
        const int num_ransac_iterations, // number of iterations to run RANSAC
        const double inlier_threshold) {
    auto clock_begin = std::chrono::steady_clock::now();

    const int num_points = input1.size(0); 

    const int num_threads_per_block = 64;
    const int num_blocks = 8;
    const int num_threads = num_blocks * num_threads_per_block;

    // CUDA Setup
    // Set GPU to use
    // int device = 0;
    cudaSetDevice(input1.get_device());

    // cudaSetDevice(input1.get_device());

    // Input data pointer (on GPU)
    double *input1_ptr = input1.data_ptr<double>();
    double *input2_ptr = input2.data_ptr<double>();

    int *num_inliers;
    double (*essential_matrices)[3][3];
    curandState* state;

    CudaErrorCheck(cudaMallocManaged((void **) &num_inliers, num_threads * sizeof(int)));
    CudaErrorCheck(cudaMallocManaged((void **) &essential_matrices, num_threads * 3 * 3 * sizeof(double)));
    CudaErrorCheck(cudaMallocManaged((void **) &state, num_threads * sizeof(curandState)));

    // Copy constants to device constant memory
    CudaErrorCheck(cudaMemcpyToSymbol(c_num_points, &num_points, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_num_test_points, &num_test_points, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_ransac_num_test_points, &num_ransac_test_points, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_ransac_num_iterations, &num_ransac_iterations, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_inlier_threshold, &inlier_threshold, sizeof(double)));

    // Generate random states, one for each thread
    SetupRandomState<<<num_blocks, num_threads_per_block>>>(seed, state);
    
    auto clock_begin_kernel = std::chrono::steady_clock::now();

    EstimateEssentialMatrix<subset_size><<<num_blocks, num_threads_per_block>>>(
                                               input1_ptr, // Two sets of matching points
                                               input2_ptr, // (flattened 2D arrays)
                                               state,      // Random number generator state
                                               num_inliers, // Number of inliers per thread
                                               essential_matrices); // Essential matrices per thread

    CudaErrorCheck(cudaPeekAtLastError()); // Check for kernel launch error
    CudaErrorCheck(cudaDeviceSynchronize()); // Check for kernel execution error

    // std::cout << "Runtime (Initialise, Kernel Only): " << std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_begin_kernel).count() << "s" << std::endl;
      
    int ind_max = distance(num_inliers, max_element(num_inliers, num_inliers + num_threads));
    // cout << "The largest element is " << ind_max << '\n';
    cout << "The number of inliers: " << num_inliers[ind_max] << '\n';

    at::Tensor E_out = at::empty(3 * 3, input1.options());
    double* dataptr = E_out.data_ptr<double>();
    CudaErrorCheck(cudaMemcpy(dataptr, &essential_matrices[ind_max], sizeof(essential_matrices[ind_max]), cudaMemcpyDeviceToDevice));

    CudaErrorCheck(cudaFree(num_inliers));
    CudaErrorCheck(cudaFree(essential_matrices));
    CudaErrorCheck(cudaFree(state));

    E_out.resize_({3, 3});

    // std::cout << "Runtime (Initialise): " << std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_begin).count() << "s" << std::endl;
    return E_out;
}


/*
 * Five point algorithm cuda initialization 
 */
std::tuple<at::Tensor, at::Tensor,int> ProjectionMatrixRansac(
        at::Tensor input1, // input 1 has size nx2, type double
        at::Tensor input2,
        const int num_test_points, // 10
        const int num_ransac_test_points, // 1000
        const int num_ransac_iterations, // number of iterations to run RANSAC
        const double inlier_threshold) {
    // auto clock_begin = std::chrono::steady_clock::now();

    const int num_points = input1.size(0); 

    const int num_threads_per_block = 64;
    const int num_blocks = 8;
    const int num_threads = num_blocks * num_threads_per_block;

    // CUDA Setup
    // Set GPU to use
    // int device = 0;
    // CudaErrorCheck(cudaSetDevice(device));

    cudaSetDevice(input1.get_device());
    // cudaSetDevice(input1.get_device());

    // Input data pointer (on GPU)
    double *input1_ptr = input1.data_ptr<double>();
    double *input2_ptr = input2.data_ptr<double>();

    int *num_inliers;
    double (*essential_matrices)[3][3];
    double (*projection_matrices)[3][4];
    curandState* state;

    CudaErrorCheck(cudaMallocManaged((void **) &num_inliers, num_threads * sizeof(int)));
    CudaErrorCheck(cudaMallocManaged((void **) &essential_matrices, num_threads * 3 * 3 * sizeof(double)));
    CudaErrorCheck(cudaMallocManaged((void **) &projection_matrices, num_threads * 3 * 4 * sizeof(double)));
    CudaErrorCheck(cudaMallocManaged((void **) &state, num_threads * sizeof(curandState)));

    // Copy constants to device constant memory
    CudaErrorCheck(cudaMemcpyToSymbol(c_num_points, &num_points, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_num_test_points, &num_test_points, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_ransac_num_test_points, &num_ransac_test_points, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_ransac_num_iterations, &num_ransac_iterations, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_inlier_threshold, &inlier_threshold, sizeof(double)));

    // Generate random states, one for each thread
    SetupRandomState<<<num_blocks, num_threads_per_block>>>(seed, state);
    
    // auto clock_begin_kernel = std::chrono::steady_clock::now();

    EstimateProjectionMatrix<subset_size><<<num_blocks, num_threads_per_block>>>(
                                               input1_ptr, // Two sets of matching points
                                               input2_ptr, // (flattened 2D arrays)
                                               state,      // Random number generator state
                                               num_inliers, // Number of inliers per thread
                                               essential_matrices,
                                               projection_matrices); // Essential matrices per thread

    CudaErrorCheck(cudaPeekAtLastError()); // Check for kernel launch error
    CudaErrorCheck(cudaDeviceSynchronize()); // Check for kernel execution error

    // std::cout << "Runtime (Initialise, Kernel Only): " << std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_begin_kernel).count() << "s" << std::endl;
      
    int ind_max = distance(num_inliers, max_element(num_inliers, num_inliers + num_threads));

    // cout << "The largest element is " << ind_max << '\n';
    // cout << "The number of inliers: " << num_inliers[ind_max] << '\n';

    at::Tensor E_out = at::empty(3 * 3, input1.options());
    at::Tensor P_out = at::empty(3 * 4, input1.options());

    
    double* dataptr = E_out.data_ptr<double>();
    double* dataptr_p = P_out.data_ptr<double>();

    CudaErrorCheck(cudaMemcpy(dataptr, &essential_matrices[ind_max], sizeof(essential_matrices[ind_max]), cudaMemcpyDeviceToDevice));
    CudaErrorCheck(cudaMemcpy(dataptr_p, &projection_matrices[ind_max], sizeof(projection_matrices[ind_max]), cudaMemcpyDeviceToDevice));

    const int Max_inlier = num_inliers[ind_max];
    E_out.resize_({3, 3});
    P_out.resize_({3, 4});

    CudaErrorCheck(cudaFree(num_inliers));
    CudaErrorCheck(cudaFree(essential_matrices));
    CudaErrorCheck(cudaFree(projection_matrices));
    CudaErrorCheck(cudaFree(state));

    // std::cout << "Runtime (Initialise): " << std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_begin).count() << "s" << std::endl;
    auto t = std::make_tuple(E_out, P_out, Max_inlier);

    return t;
}

