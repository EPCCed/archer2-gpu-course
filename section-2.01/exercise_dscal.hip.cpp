/*
 * Introduction.
 *
 * Implement the simple operation x := ax for a vector x of type double
 * and a constant 'a'.
 *
 * It introduces explicit device memory management.
 *
 * Part 1: declare and allocate device memory d_x
 * Part 2: copy host array h_x to device array d_x
 * Part 3: copy (unaltered) device array d_x back to the host array h_out
 * Part 4: remember to release device resources d_x at the end
 *
 * Training material originally developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010-2023
 */

#include <cassert>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <string>

#include "hip/hip_runtime.h"

/* Error checking routine and macro. */

__host__ void myErrorHandler(hipError_t ifail, const std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

/* The number of integer elements in the array */
#define ARRAY_LENGTH 256

/* Suggested kernel parameters */
#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 256

/* Main routine */

int main(int argc, char *argv[]) {

  size_t sz = ARRAY_LENGTH * sizeof(double);

  double a = 1.0;       /* constant a */
  double *h_x = NULL;   /* input array (host) */
  double *h_out = NULL; /* output array (host) */

  /* Check we have a GPU, and get device name from the hipDeviceProp_t
   * structure. This is for information. */

  int ndevice = 0;
  int deviceNum = -1;
  hipDeviceProp_t prop;

  HIP_ASSERT(hipGetDeviceCount(&ndevice));

  if (ndevice == 0) {
    std::cerr << "No GPU available!" << std::endl;
    std::exit(0);
  }

  HIP_ASSERT(hipGetDevice(&deviceNum));
  HIP_ASSERT(hipGetDeviceProperties(&prop, deviceNum));
  std::cout << "Device " << deviceNum << " name: " << prop.name << std::endl;
  std::cout << "Maximum number of threads per block: "
            << prop.maxThreadsPerBlock << std::endl;

  /* allocate memory on host; assign some initial values */

  h_x = new double[ARRAY_LENGTH];
  h_out = new double[ARRAY_LENGTH];
  assert(h_x);
  assert(h_out);

  for (int i = 0; i < ARRAY_LENGTH; i++) {
    h_x[i] = 1.0 * i;
    h_out[i] = 0;
  }

  /* TODO: allocate memory on device */

  /* TODO: copy input array from host to GPU */

  /* TODO: copy the result array back to the host output array */

  /* We can now check the results ... */
  std::cout << "Results:" << std::endl;
  {
    int ncorrect = 0;
    for (int i = 0; i < ARRAY_LENGTH; i++) {
      /* The print statement can be uncommented for debugging... */
      // std::cout << std::setw(9) << i << " " << std::fixed
      //           << std::setprecision(2) << std::setw(5) << h_out[i]
      //           << std::endl;
      if (fabs(h_out[i] - a * h_x[i]) < DBL_EPSILON)
        ncorrect += 1;
    }
    std::cout << "No. elements " << ARRAY_LENGTH
              << ", and correct: " << ncorrect << std::endl;
  }

  /* TODO: free device buffer */

  /* free host buffers */

  delete h_x;
  delete h_out;

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
