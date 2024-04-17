/*
 * Matrix vector product
 *
 * Here we will implement a slightly simplified version of the BLAS
 * level 2 routine dgemv() which computes the matrix-vector product
 *
 *    y_i := beta*y_i + alpha*A_ij x_j
 *
 * for an m x n matrix A_mn and vectors x_n and y_m. The data type
 * is double, with alpha and beta sclar constants.
 *
 * The simplification is that we will consider only
 *
 *    y_i := alpha*A_ij x_j
 *
 * Again we will assume that we are going to address the matrix with
 * the flattened one-dimensional index A_ij = a[i*ncol + j] with ncol
 * the number of columns n.
 *
 * An extirely serial kernel is provided below.
 *
 * Training material developed by Nick Johnson and Kevin Stratford
 * Copyright EPCC, The University of Edinburgh, 2010-2023
 */

#include <cassert>
#include <cfloat>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

#include "hip/hip_runtime.h"

__host__ void myErrorHandler(hipError_t ifail, std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

/* Kernel parameters (start with 1-d) */

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCKX 16
#define THREADS_PER_BLOCKY 16

/* An entirely serial kernel. */
__global__ void myKernel(int mrow, int ncol, double alpha, double *a, double *x,
                         double *y) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) {
    for (int i = 0; i < mrow; i++) {
      double sum = 0.0;
      for (int j = 0; j < ncol; j++) {
        sum += a[i * ncol + j] * x[j];
      }
      y[i] += alpha * sum;
    }
  }
  return;
}

/* Main routine */

int main(int argc, char *argv[]) {

  int mrow = 1024; /* Number of rows */
  int ncol = 256;  /* Number of columns (start = THREADS_PER_BLOCK) */

  double alpha = 2.0;

  double *h_x = NULL;
  double *h_y = NULL;
  double *d_x = NULL;
  double *d_y = NULL;
  double *h_a = NULL;
  double *d_a = NULL;

  /* Print device name (just for information) */

  int ndevice = 0;
  int deviceNum = -1;
  hipDeviceProp_t prop;

  HIP_ASSERT(hipGetDeviceCount(&ndevice));

  if (ndevice == 0) {
    std::cout << "No GPU available!" << std::endl;
    std::exit(0);
  }

  HIP_ASSERT(hipGetDevice(&deviceNum));
  HIP_ASSERT(hipGetDeviceProperties(&prop, deviceNum));
  std::cout << "Device " << deviceNum << " name: " << prop.name << std::endl;
  std::cout << "Maximum number of threads per block: "
            << prop.maxThreadsPerBlock << std::endl;

  /*
   * Establish some data on the host. x = 1; y = 0; A_ij = 1
   */

  h_x = new double[ncol];
  h_y = new double[mrow];
  h_a = new double[mrow * ncol];
  assert(h_x);
  assert(h_y);
  assert(h_a);

  for (int i = 0; i < ncol; i++) {
    h_x[i] = 1.0;
  }
  for (int j = 0; j < mrow; j++) {
    h_y[j] = 0.0;
  }
  for (int i = 0; i < mrow; i++) {
    for (int j = 0; j < ncol; j++) {
      h_a[i * ncol + j] = 1.0 * i * j;
    }
  }

  static __constant__ double data_read_only[3];
  double values[3] = {1.0, 2.0, 3.0};
  hipMemcpyToSymbol(data_read_only, values, 3*sizeof(double));

  /*
   * allocate memory on device
   */

  HIP_ASSERT(hipMalloc(&d_x, ncol * sizeof(double)));
  HIP_ASSERT(hipMalloc(&d_y, mrow * sizeof(double)));
  HIP_ASSERT(hipMalloc(&d_a, mrow * ncol * sizeof(double)));

  hipMemcpyKind kind = hipMemcpyHostToDevice;
  HIP_ASSERT(hipMemcpy(d_x, h_x, ncol * sizeof(double), kind));
  HIP_ASSERT(hipMemcpy(d_y, h_y, mrow * sizeof(double), kind));
  HIP_ASSERT(hipMemcpy(d_a, h_a, mrow * ncol * sizeof(double), kind));

  /* Kernel */

  uint nblockx = 1;
  dim3 blocks = {nblockx, 1, 1};
  dim3 threadsPerBlock = {THREADS_PER_BLOCK, 1, 1};

  myKernel<<<blocks, threadsPerBlock>>>(mrow, ncol, alpha, d_a, d_x, d_y);

  /* wait for all threads to complete and check for errors */

  HIP_ASSERT(hipPeekAtLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  kind = hipMemcpyDeviceToHost;
  HIP_ASSERT(hipMemcpy(h_y, d_y, mrow * sizeof(double), kind));

  std::cout << "Results:" << std::endl;
  {
    int ncorrect = 0;
    for (int i = 0; i < mrow; i++) {
      double sum = 0.0;
      double yi = 0.0;
      for (int j = 0; j < ncol; j++) {
        sum += h_a[i * ncol + j] * h_x[j];
      }
      yi = alpha * sum;
      if (fabs(yi - h_y[i]) < DBL_EPSILON)
        ncorrect += 1;
      /* Can be uncommented to debug ... */
      /* printf("Row %5d %14.7e %14.7e\n", i, yi, h_y[i]); */
    }
    std::cout << "No. rows " << mrow << ", and correct rows " << ncorrect
              << std::endl;
  }

  /* free device buffer */
  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_y));
  HIP_ASSERT(hipFree(d_x));

  /* free host buffers */
  delete h_x;
  delete h_y;
  delete h_a;

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
