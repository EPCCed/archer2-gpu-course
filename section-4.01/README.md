# Streams

HIP has the concept of independent branches of action referred to
as *streams*.

Streams are sequences of operations that are executed in order on the GPU. By using multiple streams, you can overlap operations such as kernel execution and memory transfers, leading to better utilisation of the GPU.

So far, when we execute a kernel, or a `hipMemcpy()` function, these
operations are implicitly submitted to the default stream which is
associated with the current GPU context.

Operations submitted to the same stream serialise, i.e., they are
executed in the order in which they entered the stream.

Further opportunity to overlap independent operations may be available
if we manage new, independent, streams to execute asynchronous
activities.

## Examples

Using multiple streams in GPU programming can significantly enhance performance in various scenarios. Here are some practical use cases:

1. **Overlapping Data Transfers and Computation**:
While one stream handles data transfer from the host to the device, another stream can execute a kernel. This overlap reduces idle time and maximizes GPU utilisation.

2. **Concurrent Kernel Execution**:
Running multiple independent kernels simultaneously can speed up tasks like image processing, where different filters can be applied in parallel.

3. **Asynchronous Memory Operations**:
Performing memory copies and kernel executions concurrently. For instance, copying data for the next computation while the current computation is still running.

4. **Pipeline Processing**:
In video processing, one stream can decode frames while another processes the decoded frames, and a third stream encodes the processed frames.

5. **Load Balancing**:
Distributing work across multiple streams to balance the load and avoid bottlenecks, especially in complex simulations or scientific computations.

6. **Real-Time Data Processing**:
In real-time applications like autonomous driving, one stream can handle sensor data acquisition while another processes the data for decision-making.

7. **Multi-GPU Systems**:
In systems with multiple GPUs, streams can be used to manage tasks across different GPUs, ensuring efficient resource utilisation.

Using multiple streams effectively can lead to significant performance improvements by ensuring that the GPU is always busy with useful work.

## Stream management

A stream object is declared using
```cpp
  hipStream_t stream;
```
and needs to be initialised before use via the API call
```cpp
  hipStreamCreate(&stream);
```
and is released when it is no longer required with
```cpp
  hipStreamDestroy(stream);
```
One can create an arbitrary number of streams.


### Asynchronous copies

An asynchronous form of `hipMemcpy()` is available. It has the form
```cpp
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
```cpp
  hipStreamSynchronize(stream);
```
This routine will block until all pending operations in the stream
have completed.


## Kernels

Kernels may also be submitted to a non-default stream by using an
optional argument to the execution configuration. In general, the
arguments are of the form
```cpp
  <<<dim3 blocks, dim3 threadsPerBlock, size_t shared, hipStream_t stream>>>
```
This matches the full form of the analogous `hipLaunchKernelGGL()`:
```cpp
  hipError_t hipLaunchKernelGGL(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t shared, hipStream_t stream);
```

## Page-locked, or pinned, memory

Page-locked, or pinned, memory is a type of memory that is locked in physical RAM and cannot be swapped out to disk by the operating system. This ensures that the memory remains in physical RAM, allowing for faster and more predictable access times.

HIP provides a mechanism to allow it control allocations made on the host.
So far we have used `malloc()`.

By using, e.g.,
```cpp
  double *h_ptr = NULL;
  hipHostMalloc(&h_ptr, ndata*sizeof(double));
```
in place of `malloc()` we can obtained page-locked, or pinned, memory.
By allowing HIP to supervise allocation, optimisations in transfers
may be available to the HIP run-time.

Page-locked memory should be released with
```cpp
  hipHostFree(h_ptr);
```

Such allocations are often used in conjunction with streams where
efficiency is a paramount concern.

Benefits:
1. **Faster Data Transfers**: Pinned memory allows for faster data transfers between the host (CPU) and the device (GPU) because it avoids the overhead of paging.
2. **Predictable Performance**: Since the memory is always resident in RAM, access times are more predictable, which is crucial for real-time applications.
3. **Direct Access by GPU**: The GPU can directly access pinned memory without involving the CPU, enabling more efficient data transfers.

However, pinned memory is more limited than pageable memory, so it should be used judiciously.

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
   performance, it should be possible to view the result from `rocprof` with a
   tool like [Perfetto](https://ui.perfetto.dev/) and see the different streams
   in operation.

2. Check you can replace the host allocations of `x` and `y` with
   `hipHostMalloc()` and make the appropriate adjustment to free
   resources at the end of execution.

<!-- ### Finished?

(No equivalent in HIP at the moment)

Note that it is possible to add a meaningful label to a stream
(and to other types of object) via the NVTX library. The label
will then appear in the Nsight profile. For a stream use:
```
  void nvtxNameCudaStreamA(cudaStream_t stream, const char * name);
```
to attach an ASCII label to a stream. -->
