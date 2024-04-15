# Streams

HIP has the concept of independent branches of action referred to
as *streams*.

So far, when we execute a kernel, or a `hipMemcpy()` function, these
operations are implicitly submitted to the default stream which is
associated with the current GPU context.

Operations submitted to the same stream serialise, i.e., they are
executed in the order in which they entered the stream.

Further opportunity to overlap independent operations may be available
if we manage new, independent, streams to execute asynchronous
activities.

## Stream management

A stream object is declared using
```
  hipStream_t stream;
```
and needs to be initialised before use via the API call
```
  hipStreamCreate(&stream);
```
and is released when it is no longer required with
```
  hipStreamDestroy(stream);
```
One can create an arbitrary number of streams.


### Asynchronous copies

An asynchronous form of `hipMemcpy()` is available. It has the form
```
  hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sz,
                            hipMemcpyKind kind, hipStream_t stream);
```
If the final stream argument is omitted, the operation uses the default
stream.

Note this still uses both host and device memory references like
`hipMemcpy()` and unlike `hipMemPrefetchAsync()`.

### Synchronisation

To know when an asynchronous stream operation can be considered
complete, and that it is safe to make use of the result, we need
to synchronise.
```
  hipStreamSynchronize(stream);
```
This routine will block until all pending operations in the stream
have completed.


## Kernels

Kernels may also be submitted to a non-default stream by using an
optional argument to the execution configuration. In general, the
arguments are of the form
```
  <<<dim3 blocks, dim3 threadsPerBlock, size_t shared, hipStream_t stream>>>
```
This matches the full form of the analogous `hipLaunchKernel()`:
```
  hipError_t hipLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t shared, hipStream_t stream);
```

## Page-locked, or pinned, memory

HIP provides a mechanism to allow it control allocations made on the host.
So far we have used `malloc()`.

By using, e.g.,
```
  double *h_ptr = NULL;
  hipHostMalloc(&h_ptr, ndata*sizeof(double));
```
in place of `malloc()` we can obtained page-locked, or pinned, memory.
By allowing HIP to supervise allocation, optimisations in transfers
may be available to the HIP run-time.

Page-locked memory should be released with
```
  hipFreeHost(h_ptr);
```

Such allocations are often used in conjunction with streams where
efficiency is a paramount concern.


## Exercise

Revisit the previous problem for BLAS call `dger()`. (A new working
template is supplied.)

The exercise is just to illustrate the use of streams, and of
page-locked host memory.

Suggested procedure:

1. For vectors `x` and `y` replace the relevant `hipMemcpy()` with
   an asynchronous operation using two different streams. Make sure
   that the data has reached the device before the kernel launch.

   While it is unlikely that this will have any significant beneficial effect in
   performance, it should be possible to view the result from rocprof with a
   tool like [Perfetto](https://ui.perfetto.dev/) and see the different streams
   in operation.

2. Check you can replace the host allocations of `x` and `y` with
   `hipHostMalloc()` and make the appropriate adjustment to free
   resources at the end of execution.

<!-- ### Finished?

(No qeqivalent in HIP at the moment)

Note that it is possible to add a meaningful label to a stream
(and to other types of object) via the NVTX library. The label
will then appear in the Nsight profile. For a stream use:
```
  void nvtxNameCudaStreamA(cudaStream_t stream, const char * name);
```
to attach an ASCII label to a stream. -->
