# Some performance considerations

We should now have a functioning, if simple, GPU program which moves
some data to and from the device and executes a kernel.

We can now ask the question: what factors can influence the
performance of such a program?


## Parallelism

Amdahl's law states that the parallel performance of a program is
limited by the fraction of code that is serial.

In the GPU context, this has two manifestations:

1. Kernel code is potentially parallel
2. Host code is definitely serial (including host-device transfers)

This may mean that additional work is required to expose parallelism,
or eliminate host operations in favour of device operations.

### Occupancy

Potentially, the GPU has a lot of CUs/cores that can be used. Having very
many blocks of work available at an one time is said to favour
high *occupancy*.

Occupancy is one of the key concepts that all GPU programmers should
be aware of. Profilers can provide complete information regarding the limiting
factors of a kernel's occupancy (see the
[profiling](../section-3.01/README.md#profiling) section). Basic knowledge of
how to tune the parameters that influence occupancy can lead to easy and
dramatic performance improvements. The only thing that is likely to have a
greater influence on performance is using sensible memory access patterns
(see this [section](#memory-usage)).

Occupancy may be thought of simply as having a very high degree of thread
parallelism. However, the degree is much higher than would be expected
on the basis of a threaded CPU program (where threads is usually the
number of cores).

Typically, a GPU kernel wants at least O(10^5) or O(10^6) threads to be
effective. That is, the problem space should have this number of elements.

If the problem does not have this natural degree of (data) parallelism,
there may be little benefit in using a GPU.

### Two dimensional example

Consider a two-dimensional loop:
```cpp
   int NX = 512;
   int NY = 512;
   ...
   for (int i = 0; i < NX; i++) {
     for (int j = 0; j < NY; j++) {
       /* ... work for element (i, j) ... */
     }
   }
```

If we parallelised the inner loop only, we would have work for at most
512 threads. This would be two blocks if we were using, e.g., 256 threads
per blocks.

This would clearly be poor occupancy.


If we parallelised both loops, we would have 512 x 512 = 262,144 threads
(1024 blocks). This is much better. We now have a chance to employ many
CUs.

Occupancy depends on two other constraints beyond utilising all threads in a
CU: thread-local register usage and shared memory usage.

### Thread-local memory and occupancy

As well as being designed for a high degree of parallelisation, GPUs are most
efficient for problems that have a lot of computation performed in each thread.
For this reason CUs are allocated a lot of thread-local registers which are
the fastest form of memory. Despite this, registers are still a limited
resource and it is very common to run out of registers before you run out of
threads within a CU.

You have already been introduced to an example of thread-local memory in
[this](../section-2.02/README.md#a-simple-example) example where you have:

```cpp
  __global__ void myKernel(int *result) {

    int i = threadIdx.x;
```

here the variable `i` is thread-local and will normally be stored in a
register.

If you have used a profiler to identify that register usage is limiting
occupancy there are several ways that you can reduce the amount of registers to
boost occupancy and performance. Some ways are easy and obvious. For example,
it is good practice to only declare a variable near to where it is being used.
In some cases choosing a smaller block size will allow a CU to manage more
threads simultaneously (and hence have higher occupancy) because the number of
registers can be closer to the limit without breaching it: Consider the case
that you have enough registers only for around 400 threads to run on a single
CU. If you choose a block size of 256 threads you will only be able to fit one
block on the CU. However switching to a block size of 128 threads will allow
you to fit three blocks of 128 threads, significantly boosting the occupancy;
providing that no other constraints apply: this brings us to shared memory!

### Shared memory and occupancy

"Shared memory" is an important concept in GPU programming because many important
computational patterns are not "embarrassingly parallel": they require some
degree of inter-thread communication. Shared memory allows this
for threads within the same block. More complete details of shared memory will
be provided in a later [section](../section-2.05/README.md#shared-memory).
shared memory can be declared within a kernel body in the following way:

```cpp
  __global__ void myKernel(int *result) {
__shared__ double sharedArray[8];
```
In the above example the shared memory is statically allocated at compile time.
Like registers (and threads), shared memory is a finite resource that can limit
occupancy.
On AMD GPUs the amount of shared memory is a fixed quantity and considerations
of occupancy follow the logic outlined in the previous section for registers.
For Nvidia GPUs it is also important for programmers to be aware that the
available amount of shared memory is something that can be dynamically tuned at
runtime. This is because shared memory resides within the L1 cache (used to
cache global memory within the CU) in modern Nvidia GPUs. It is possible to
"carveout" a larger amount of shared memory at the price of reducing the L1
cache. For details of this refer to Nvidia documentation. If shared memory is
not a limiting resource, then you can just rely on statically allocated shared
memory.

It is also worth programmers being aware that they can choose to replace
thread-local variables with shared variables if they are running out of
registers but still have plenty of shared memory. Whilst shared memory is not
designed to be used in such a thread-local way and is not as fast as registers,
sometimes this cost can be significantly outweighed by an increase in
occupancy.

Sometimes the compiler register allocation heuristics will also prefer to use
global memory for thread-local variables instead of running out of registers.
Using global memory for thread-local memory operations is extremely slow. Since
CUDA version 13.0 Nvidia introduced an option to spill registers to shared
memory instead of global memory, which can be a good option if the compiler is
spilling registers and you have plenty of shared memory resources left over.
For further information see this Nvidia
[blog](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/).


### Final remarks on occupancy

Whilst the GPU's CUs can manage more blocks than you are ever likely to have to
worry about, the number of wavefronts that the CUs of modern Nvidia/AMD GPUs
can simultaneously execute is four. For this reason it is a good idea to try to
make it possible to have a minimum block size of 128 threads for Nvidia GPUs
and AMD (e.g. RDNA) workstation GPUs that have 32 threads per wave-front, or
256 for AMD CDNA GPUs which have 64 threads per wave-front. Another factor to
consider is that the block size should also be a multiple of the number of
threads in a wavefront, since threads within a wavefront execute in lock-step.

**Expert point**. For completeness there is one more factor to be aware of.
Optimal register allocation (and register spilling) is an NP-hard problem that
is inherently linked with ideal block-size and occupancy. Because it is NP-hard
compilers must rely on in-exact heuristics to choose how to allocate or spill
registers. However it is likely that your GPU compiler will pick good register
allocation for most codebases. However if you have a particularly complex
codebase there are manual knobs which you can tweak in source code such as the
degree that functions are inlined or that loops are unrolled, which influence
register allocation. Under normal circumstances these considerations can be
left till the very end of program design/optimisation. Conversely the compiler
can occasionally be too aggressive in spilling registers which can often be
even worse than a drop in occupancy. Achieving good improvements usually
requires a careful examination of the "assembly" generated by the compiler and
is beyond the scope of this course.

In summary, it is vital to be aware of the limited register and shared memory
resources available to you when designing your kernels. This section should
equip you well with the knowledge that when combined with the profiler
know-how that you will learn in the
[profiling](../section-3.01/README.md#profiling) section, will allow you to
maximise occupancy when running your kernels.

## Memory usage

### CPU: caching behaviour

A given thread in a CPU code favours consecutive memory accesses.
E.g., in C, recall that it is the right-most index that runs
fastest in memory.
```cpp
   for (int i = 0; i < NX; i++) {
     for (int j = 0; j < NY; j++) {
       a[i][j] = 0.0;
     }
   }
```
Such an order displays favourable cache behaviour. A single thread makes
contiguous memory accesses.


### GPU: coalescing behaviour

For GPU global memory, the opposite is true. The hardware wants
to have waveforms of consecutive threads load consecutive memory
locations in a contiguous block.

Consider a one-dimensional example:
```cpp
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  a_output[i] = a_input[i];
```
Here, there is no issue, consecutive threads (those with consecutive
x-index) access consecutive memory locations.


### Two dimensions again

Consider first:
```cpp
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  for (int j = 0; j < NY; j++) {
    a_output[i][j] = a_input[i][j];
  }
```
Here, a given thread makes `NY` consecutive accesses to the arrays. This
does not favour coalesced access.

We want consecutive threads to have consecutive accesses, e.g.,
```cpp
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i = 0; i < NX; i++) {
    a_output[i][j] = a_input[i][j];
  }
```

This is the favoured pattern. In other words, coalescing favours a given
thread making a strided memory access.


## Exercise (30 minutes)

The following exercise will examine the issue of parallelism and occupancy.
The current directory is a template `exercise_dger.hip.cpp` in which you
are asked to implement a kernel which computes the following matrix
operation
$$A_{ij} = A_{ij} + \alpha x_i y_j$$
for a matrix $A$ with `m` rows and `n` columns, a vector $x$ of length `m`, a
vector $y$ of length `n`, and constant $\alpha$. The data type is
`double` in all cases.

For the matrix $A$, we will adopt a flattened one-dimensional indexing
for which element row `i` and column `j` is addressed as `a[i*ncol + j]`.

As this is partly a performance issue (a correct answer is also required!),
we will implement some simple profiling by adding `rocprof` to the submission
script.

`rocprof` gives some basic text-based profile information for routines involving
the GPU at the end of execution. Try to keep a note of the time taken by the
kernel at each stage (reported in nanoseconds, `ns` by `rocprof`).


A suggested procedure is:
1. Check the template to see that the matrix and vectors have been established
   in device memory. Note that the template uses the HIP API call
   ```cpp
      hipError_t hipMemset(void * dptr, int value, size_t sizeBytes);
   ```
   to initialise all the device matrix elements to zero directly. The template
   should compile and run, but will not compute the correct answer as the
   kernel stub supplied does nothing.
2. Implement the most simple kernel in which the update is entirely
   serialised. E.g.,
   ```cpp
   int tid = blockIdx.x*blockDim.x + threadIdx.x;

   if (tid == 0) {
     for (int i = 0; i < mrow; i++) {
       for (int j = 0; j < ncol; j++) {
          a[ncol*i + j] = a[ncol*i + j] + alpha*x[i]*y[j];
        }
     }
   }
   ```
   Check the execution configuration and run the code to ensure it reports
   the correct answer.

3. Eliminate the `i`-loop and re-check the kernel launch parameters to
   provide parallelism over rows of the matrix.
   Remember to allow that the problem size is not a whole number
   of blocks.

4. In addition, eliminate the `j`-loop to have parallelism over
   both rows and columns. You will need to introduce two dimensions
   in the abstract description, e.g., via
   ```cpp
   int j = blockIdx.y*blockDim.y + threadIdx.y;
   ```
   and make an appropriate adjustment to the kernel launch parameters.
   Hint: keep the same total number of threads per block; but the block
   must become two-dimensional.

5. Is your resultant code getting the coalescing right? Consecutive
   threads, that is, threads with consecutive $x$-index, should
   access consecutive memory location.



### Finished?

If we had not used `hipMemset()` to initialise the device values for
the matrix, what other options to initialise these values on the device
are available to us? `hipMemset()` is limited in that it can only be
used to initialise array values to zero, but not to other, non-zero, values.

For your best effort for the kernel, what is the overhead of the actual kernel
launch (`hipLaunchKernel` in the profile) compared with the time taken for the
kernel itself? These can be found in `results.stats.csv` and
`results.hip_stats.csv`, or in `results.json`.

What's the overhead for the host-device transfers?
