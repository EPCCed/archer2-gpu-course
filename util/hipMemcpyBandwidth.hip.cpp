#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "hip/hip_runtime.h"

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#define omp_get_wtime() ((double)clock() * (1.0 / CLOCKS_PER_SEC))
#define omp_get_wtick() (1.0 / CLOCKS_PER_SEC)
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

/* Error checking routine and macro. */

__host__ void myErrorHandler(hipError_t ifail, std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

/* Single device buffer */

struct buffer_t {
  size_t sz;
  double *data_d;
  double *data_h;
};

int bufferAllocate(int ndouble, buffer_t *buf) {

  assert(buf);

  buf->sz = sizeof(double) * ndouble;

  HIP_ASSERT(hipMalloc((void **)&buf->data_d, buf->sz));

  buf->data_h = (double *)calloc(ndouble, sizeof(double));
  assert(buf->data_h);

  return 0;
}

int bufferFree(buffer_t *buf) {

  assert(buf);

  HIP_ASSERT(hipFree(buf->data_d));
  free(buf->data_h);

  *buf = (buffer_t){0};

  return 0;
}

int peerAccess(int ndevice) {

  /* Matrix should be symmetric */
  /* hipDeviceCanAccessPeer(&flag, fromWhich, onWhichMemoryResides) */

  printf("Peer Access:\n");
  printf("     ");
  for (int ia = 0; ia < ndevice; ia++) {
    printf(" %3d ", ia);
  }
  printf("\n");

  for (int ia = 0; ia < ndevice; ia++) {
    printf(" %2d  ", ia);
    for (int ib = 0; ib < ndevice; ib++) {
      if (ia == ib) {
        printf(" n/a ");
      } else {
        int flag = -1;
        HIP_ASSERT(hipDeviceCanAccessPeer(&flag, ia, ib));
        printf(" %3d ", flag);
      }
    }
    printf("\n");
  }

  return 0;
}

int peerAccessUpdate(int ndevice, int enable) {

  /* In both directions */

  for (int ia = 0; ia < ndevice; ia++) {
    HIP_ASSERT(hipSetDevice(ia));
    for (int ib = 0; ib < ndevice; ib++) {
      if (ia == ib) {
        continue;
      } else {
        int flag = -1;
        HIP_ASSERT(hipDeviceCanAccessPeer(&flag, ia, ib));
        if (flag && enable == 1)
          hipDeviceEnablePeerAccess(ib, 0);
        if (flag && enable == 0)
          hipDeviceDisablePeerAccess(ib);
      }
    }
  }

  return 0;
}

int bandwidthHostToDevice(int nnuma, int ndevice, size_t ndouble,
                          hipMemcpyKind kind) {

  buffer_t *abuf = {0}; /* Array of buffers */

  assert(kind != hipMemcpyDeviceToDevice);

  abuf = (buffer_t *)calloc(ndevice, sizeof(buffer_t));
  assert(abuf);

  /* Establish buffers per device */
  for (int n = 0; n < ndevice; n++) {
    HIP_ASSERT(hipSetDevice(n));
    bufferAllocate(ndouble, abuf + n);
  }

  printf("\n");
  if (kind == hipMemcpyHostToDevice)
    printf("memcpyHostToDevice (B/s)\n");
  if (kind == hipMemcpyDeviceToHost)
    printf("memcpyDeviceToHost (B/s)\n");
  printf("Message size: %ld B\n", ndouble * sizeof(double));
  printf("\n");

  /* Table heading */

  for (int id = 0; id < ndevice; id++) {
    if (id == 0)
      printf("%6s", "");
    printf("%8s%1d", "GPU", id);
    if (id == ndevice - 1)
      printf("\n");
  }

  for (int numa = 0; numa < nnuma; numa++) {
    printf("%4s%1d  ", "numa", numa);
    for (int id = 0; id < ndevice; id++) {
      int nrephost = 100;
      buffer_t *b1 = abuf + id;
      double t0 = 0.0;
      HIP_ASSERT(hipSetDevice(id));
      t0 = omp_get_wtime();
      if (kind == hipMemcpyHostToDevice) {
        for (int nrep = 0; nrep < nrephost; nrep++) {
          HIP_ASSERT(hipMemcpy(b1->data_d, b1->data_h, b1->sz, kind));
        }
      } else {
        for (int nrep = 0; nrep < nrephost; nrep++) {
          HIP_ASSERT(hipMemcpy(b1->data_h, b1->data_d, b1->sz, kind));
        }
      }
      /* Time and then B/s */
      t0 = (omp_get_wtime() - t0) / nrephost;
      printf("%8.2e ", b1->sz / t0);
    }
    printf("\n");
    /* Next numa */
  }

  /* Free buffers */

  for (int n = 0; n < ndevice; n++) {
    HIP_ASSERT(hipSetDevice(n));
    bufferFree(abuf + n);
  }

  return 0;
}

int bandwidthDeviceToDevice(int ndevice, int ndouble, int nrepeats) {

  buffer_t *abuf = {0};

  abuf = (buffer_t *)calloc(ndevice, sizeof(buffer_t));
  assert(abuf);

  /* Establish buffers per device */
  for (int n = 0; n < ndevice; n++) {
    HIP_ASSERT(hipSetDevice(n));
    bufferAllocate(ndouble, abuf + n);
  }

  printf("\n");
  printf("memcpyDeviceToDevice (B/s)\n");
  printf("Message size: %ld B\n", ndouble * sizeof(double));
  printf("\n");

  /* Table heading (b2 is the src) */

  for (int id = 0; id < ndevice; id++) {
    if (id == 0)
      printf("%6s", "");
    printf("%8s%1d", "GPU", id);
    if (id == ndevice - 1)
      printf("\n");
  }

  for (int n1 = 0; n1 < ndevice; n1++) {
    HIP_ASSERT(hipSetDevice(n1));
    printf("%4s%1d  ", "GPU", n1);
    for (int n2 = 0; n2 < ndevice; n2++) {

      if (n1 == n2) {
        /* Could copy from one buffer to another on same device */
        printf("%9s ", "n/a");
      } else {
        /* b1 is the destination */
        buffer_t *b1 = abuf + n1;
        buffer_t *b2 = abuf + n2;
        double t0 = 0.0;
        t0 = omp_get_wtime();

        for (int nrep = 0; nrep < nrepeats; nrep++) {
          HIP_ASSERT(hipMemcpy(b1->data_d, b2->data_d, b1->sz,
                                 hipMemcpyDeviceToDevice));
          HIP_ASSERT(hipDeviceSynchronize());
        }
        /* Time and then B/s */
        t0 = (omp_get_wtime() - t0) / nrepeats;
        printf("%8.2e ", b1->sz / t0);
      }
      /* Next n2 */
    }
    printf("\n");
  }

  /* Free buffers */

  for (int n = 0; n < ndevice; n++) {
    HIP_ASSERT(hipSetDevice(n));
    bufferFree(abuf + n);
  }

  return 0;
}

int main(int argc, char **argv) {

  int ndevice = 0;
  int nnuma = 0;
  size_t ndouble = 1024 * 1024 * 8; /* Msg size: number of doubles */

  HIP_ASSERT(hipGetDeviceCount(&ndevice));
  nnuma = omp_get_max_threads();
  printf("Device count:          %2d\n", ndevice);
  printf("omp_get_max_threads(): %2d\n", nnuma);

  bandwidthHostToDevice(nnuma, ndevice, ndouble, hipMemcpyHostToDevice);
  bandwidthHostToDevice(nnuma, ndevice, ndouble, hipMemcpyDeviceToHost);
  bandwidthHostToDevice(nnuma, ndevice, ndouble, hipMemcpyHostToDevice);
  bandwidthHostToDevice(nnuma, ndevice, ndouble, hipMemcpyDeviceToHost);

  peerAccessUpdate(ndevice, 0);
  peerAccess(ndevice);
  bandwidthDeviceToDevice(ndevice, ndouble, 50);

  /* Peer access */

  peerAccess(ndevice);
  peerAccessUpdate(ndevice, 1);

  bandwidthDeviceToDevice(ndevice, ndouble, 100);

  printf("Complete\n");

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