/*
 * Use of shared memory and atomic updates.
 *
 * An implementation of the blas-like level 1 routine ddot(), which is
 * the vector scalar product
 *
 *   res = x_n y_n
 *
 * where x and y are both vectors (of type double) and of length n elements.
 *
 *
 * Copyright EPCC, The University of Edinburgh, 2023
 */

#include <cassert>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <string>

#include "hip/hip_runtime.h"

__host__ void myErrorHandler(hipError_t ifail, std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

/* Kernel parameters */

#define THREADS_PER_BLOCK 256

__global__ void ddot(int n, double *x, double *y, double *result) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
      sum += x[i] * y[i];
    }
    *result = sum;
  }

  return;
}

/* Main routine */

int main(int argc, char *argv[]) {

  int nvector = 2048 * 64; /* Length of vectors x, y */

  double *h_x = NULL;
  double *h_y = NULL;
  double *d_x = NULL;
  double *d_y = NULL;

  double h_result = 0.0;
  double *d_result = NULL;

  /* Establish host data (with some initial values for x and y) */

  h_x = new double[nvector];
  h_y = new double[nvector];
  assert(h_x);
  assert(h_y);

  for (int i = 0; i < nvector; i++) {
    h_x[i] = 1.0 * i;
  }
  for (int j = 0; j < nvector; j++) {
    h_y[j] = 2.0 * j;
  }

  /* Establish device data and initialise A to zero on the device */
  /* Copy the initial values of x and y to device memory */
  /* Also need device memory for the (scalar) result */

  HIP_ASSERT(hipMalloc(&d_x, nvector * sizeof(double)));
  HIP_ASSERT(hipMalloc(&d_y, nvector * sizeof(double)));

  hipMemcpyKind kind = hipMemcpyHostToDevice;
  HIP_ASSERT(hipMemcpy(d_x, h_x, nvector * sizeof(double), kind));
  HIP_ASSERT(hipMemcpy(d_y, h_y, nvector * sizeof(double), kind));

  HIP_ASSERT(hipMalloc(&d_result, sizeof(double)));

  /* Define the execution configuration and run the kernel */

  uint nblockx = 1 + (nvector - 1) / THREADS_PER_BLOCK;
  dim3 blocks = {nblockx, 1, 1};
  dim3 threadsPerBlock = {THREADS_PER_BLOCK, 1, 1};

  ddot<<<blocks, threadsPerBlock>>>(nvector, d_x, d_y, d_result);

  HIP_ASSERT(hipPeekAtLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  /* Retrieve the result and check. */

  kind = hipMemcpyDeviceToHost;
  HIP_ASSERT(hipMemcpy(&h_result, d_result, sizeof(double), kind));

  double result = 0.0;
  for (int i = 0; i < nvector; i++) {
    result += h_x[i] * h_y[i];
  }
  std::cout << std::setprecision(8);
  std::cout << "Result for device dot product is: " << h_result << " (correct "
            << result << ")" << std::endl;
  std::cout << std::defaultfloat;
  if (fabs(h_result - result) < DBL_EPSILON) {
    std::cout << "Correct" << std::endl;
  } else {
    std::cout << "FAIL!" << std::endl;
  }

  /* Release resources */

  HIP_ASSERT(hipFree(d_x));
  HIP_ASSERT(hipFree(d_y));
  HIP_ASSERT(hipFree(d_result));
  delete h_y;
  delete h_x;

  return 0;
}

/* It is important to check the return code from API calls, so the
 * follow function/macro allow this to be done concisely as
 *
 *   HIP_ASSERT(hipRunTimeAPIFunction(...));
 *
 * Return codes may be asynchronous, and thus misleading! */

__host__ void myErrorHandler(hipError_t ifail, const std::string file, int line,
                             int fatal) {

  if (ifail != hipSuccess) {
    std::cerr << "Line " << line << " (" << file
              << "): " << hipGetErrorName(ifail) << ": "
              << hipGetErrorString(ifail) << std::endl;
    if (fatal)
      std::exit(ifail);
  }

  return;
}
