#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>

#include "hip/hip_runtime.h"

#define HAVE_GPU_AWARE_MPI 1

__host__ void myErrorHandler(hipError_t ifail, std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

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

/*****************************************************************************
 *
 *  bandwidthDeviceToDevice
 *
 *****************************************************************************/

int bandwidthDeviceToDevice(MPI_Comm comm, int ndouble, int nrepeats) {

  int nrank = -1;
  int rank = -1;
  double t_host = -1.0; /* Time for transfer (s) */
  double t_device = -1.0;

  buffer_t send = {0};
  buffer_t recv = {0};

  MPI_Comm_size(comm, &nrank);
  MPI_Comm_rank(comm, &rank);

  bufferAllocate(ndouble, &send);
  bufferAllocate(ndouble, &recv);

  if (HAVE_GPU_AWARE_MPI) {

    t_device = MPI_Wtime();

    for (int irep = 0; irep < nrepeats; irep++) {

      int isrc = irep % 2;
      int idst = (irep + 1) % 2;
      int tag = 220728;

      if (rank == isrc) {
        MPI_Send(send.data_d, ndouble, MPI_DOUBLE, idst, tag, comm);
      } else {
        MPI_Recv(recv.data_d, ndouble, MPI_DOUBLE, isrc, tag, comm,
                 MPI_STATUS_IGNORE);
      }
    }

    /* Time */
    t_device = (MPI_Wtime() - t_device) / nrepeats;
  }

  /* Copy via host (always available) */

  t_host = MPI_Wtime();

  for (int irep = 0; irep < nrepeats; irep++) {

    int isrc = irep % 2;
    int idst = (irep + 1) % 2;
    int tag = 220728;

    if (rank == isrc) {
      size_t sz = sizeof(double) * ndouble;
      hipMemcpyKind kind = hipMemcpyDeviceToHost;
      HIP_ASSERT(hipMemcpy(send.data_h, send.data_d, sz, kind));
      MPI_Send(send.data_h, ndouble, MPI_DOUBLE, idst, tag, comm);
    } else {
      size_t sz = sizeof(double) * ndouble;
      hipMemcpyKind kind = hipMemcpyHostToDevice;
      MPI_Recv(recv.data_h, ndouble, MPI_DOUBLE, isrc, tag, comm,
               MPI_STATUS_IGNORE);
      HIP_ASSERT(hipMemcpy(recv.data_d, recv.data_h, sz, kind));
    }

    /* next repeat */
  }

  t_host = (MPI_Wtime() - t_host) / nrepeats;

  /* Report */

  if (rank == 0) {
    double bs_host = send.sz / t_host;
    double bs_device = send.sz / t_device;
    if (HAVE_GPU_AWARE_MPI == 0)
      bs_device = 0.0;
    printf("%14ld %9.3e %9.3e %9.3e %9.3e\n", send.sz, t_host, bs_host,
           t_device, bs_device);
  }

  bufferFree(&recv);
  bufferFree(&send);

  return 0;
}

int main(int argc, char **argv) {

  int ndevice = 0;
  int nrank = -1;
  int rank = -1;
  size_t nmax = 1024 * 1024 * 128; /* Msg size: number of doubles */

  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(comm, &nrank);
  MPI_Comm_rank(comm, &rank);

  HIP_ASSERT(hipGetDeviceCount(&ndevice));

  if (rank == 0) {
    printf("Device count:          %2d\n", ndevice);
    printf("MPI_Comm_size():       %2d\n", nrank);
    printf("\n");
    printf("%14s %9s %9s %9s %9s\n", "Message (B)", "secs", "B/s", "secs",
           "B/s");
    printf("%14s %9s %9s %9s %9s\n", "", "host", "host", "device", "device");
  }

  /* Just set GPU == MPI rank in this simple case */
  assert(rank < ndevice);
  HIP_ASSERT(hipSetDevice(rank));

  for (size_t nsz = 1; nsz < nmax; nsz *= 2) {
    bandwidthDeviceToDevice(comm, nsz, 50);
  }

  if (rank == 0)
    printf("Complete\n");

  MPI_Finalize();

  return 0;
}
