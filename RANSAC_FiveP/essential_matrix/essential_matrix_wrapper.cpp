#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <iostream>

// CUDA forward declarations

at::Tensor EssentialMatrixInitialise(
    at::Tensor input1, // point set 1, dimension nx2
    at::Tensor input2, // point set 2, dimension nx2
    const int num_test_points, // 10
    const int num_ransac_test_points, // 1000
    const int num_ransac_iterations, // number of iterations to run RANSAC
    const double inlier_threshold);

std::tuple<at::Tensor, at::Tensor,int> ProjectionMatrixRansac(
    at::Tensor input1, // point set 1, dimension nx2
    at::Tensor input2, // point set 2, dimension nx2
    const int num_test_points, // 10
    const int num_ransac_test_points, // 1000
    const int num_ransac_iterations, // number of iterations to run RANSAC
    const double inlier_threshold);

at::Tensor EssentialMatrixOptimise(
        at::Tensor input1,
        at::Tensor input2,
        at::Tensor E_init,
        const double delta,
        const double alpha,
        int MaxReps);
    
at::Tensor EssentialMatrixDecompose(
    at::Tensor Emat);

std::tuple<at::Tensor, at::Tensor> EssentialMatrixDecomposeUV(
    at::Tensor Emat);
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_DOUBLE(x) TORCH_CHECK(x.scalar_type()==at::ScalarType::Double, #x, " must be a double tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT_INIT(x) CHECK_CUDA(x); CHECK_DOUBLE(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_OPT(x) CHECK_DOUBLE(x); CHECK_CONTIGUOUS(x)

at::Tensor EssentialMatrixInitialiseWrapper(
    at::Tensor input1,
    at::Tensor input2,
    const int num_test_points,
    const int num_ransac_test_points,
    const int num_ransac_iterations,
    const double inlier_threshold) {

    CHECK_INPUT_INIT(input1);
    CHECK_INPUT_INIT(input2);

    return EssentialMatrixInitialise(input1, input2, num_test_points, num_ransac_test_points, num_ransac_iterations, inlier_threshold);
}    

std::tuple<at::Tensor, at::Tensor,int> ProjectionMatrixRansacWrapper(
    at::Tensor input1,
    at::Tensor input2,
    const int num_test_points,
    const int num_ransac_test_points,
    const int num_ransac_iterations,
    const double inlier_threshold) {

    CHECK_INPUT_INIT(input1);
    CHECK_INPUT_INIT(input2);

    return ProjectionMatrixRansac(input1, input2, num_test_points, num_ransac_test_points, num_ransac_iterations, inlier_threshold);
}    

at::Tensor EssentialMatrixOptimiseWrapper(
    at::Tensor input1,
    at::Tensor input2,
    at::Tensor E_init,
    const double delta,
    const double alpha,
    const int MaxReps) {

    // newly added, not yet tested
    CHECK_INPUT_OPT(input1)
    CHECK_INPUT_OPT(input2)
    CHECK_INPUT_OPT(E_init)

    return EssentialMatrixOptimise(input1, input2, E_init, delta, alpha, MaxReps);
}    

at::Tensor EssentialMatrixDecomposeWrapper(
    at::Tensor Emat) {
    return EssentialMatrixDecompose(Emat);
}

std::tuple<at::Tensor, at::Tensor> EssentialMatrixDecomposeUVWrapper(
    at::Tensor Emat) {
    return EssentialMatrixDecomposeUV(Emat);
}


// if we have grad here

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialise", &EssentialMatrixInitialiseWrapper, "Initialise essential matrix using RANSAC-5pt");
    m.def("optimise", &EssentialMatrixOptimiseWrapper, "Optimise essential matrix using iteratively reweighted least squares");
    m.def("computeP", &ProjectionMatrixRansacWrapper, "Compute E and P matrix using RANSAC-5pt");
    m.def("decompose", &EssentialMatrixDecomposeWrapper, "Decompose essential matrix into angles");
    m.def("decomposeUV", &EssentialMatrixDecomposeUVWrapper, "Decompose essential matrix into U, V. (E = UI^V', I^=diag(1, 1, 0).)");
}
