# Managed memory

HIP provides a number of different ways to establish device
memory and transfer data between host and device.

Different mechanisms may be favoured in different situations.

## Explicit memory allocation/copy

We have seen the explicit mechanics of using standard C pointers.
Schematically:

```cpp
  double *h_ptr = NULL;
  double *d_ptr = NULL;

  h_ptr = (double *)malloc(nbytes);

  hipMalloc(&d_ptr, nbytes);
  hipMemcpy(d_ptr, h_ptr, nbytes, hipMemcpyHostToDevice);
```

The host pointer to the device memory is then used in the kernel invocation.

```cpp
  myKernel<<<...>>>(d_ptr);
```

However, pointers to device memory cannot be dereferenced on the host.

This is a perfectly sound mechanism, particularly if we are only
considering the transfers of large contiguous blocks of data.
(It is also likely to be the fastest mechanism.)

However, this can become onerous if there are complex data access
patterns, or if rapid testing and development are required. It also
gives rise to the need to have both a host reference and a device
reference in the code (`h_ptr` and `d_ptr`).

## Managed memory (or unified memory)

Managed memory is allocated on the host via

```cpp
__host__ hipError_t hipMallocManaged(void **ptr, size_t sz);
```

in place of the combination of `malloc()` or `new` (in C++) and `hipMalloc()`.

This establishes an effective single reference to memory which can be
accessed on both host and device.

Host/device transfers are managed automatically as the need arises.

So, a schematic of usage might be:

```cpp
  double *ptr = NULL;

  hipMallocManaged(&ptr, nbytes);

  /* Initialise values on host ... */

  for (int i = 0; i < ndata; i++) {
    ptr[i] = 1.0;
  }

  /* Use data in a kernel ... */
  kernel<<<...>>>(ptr);
```

### Releasing managed memory

Managed memory established with `hipMallocManaged()` is released with

```cpp
  hipFree(ptr);
```

which is the same as for memory allocated via `hipMalloc()`.

### Mechanism: page migration

Transfers are implemented through the process of page migration.
A page is the smallest unit of memory management and is often
4096 bytes on a typical (CPU) machine.

Assume - and this may or may not be the case - that
`hipMallocManaged()` establishes memory in the host space.
We can initialise memory on the host and call a kernel.

When the GPU starts executing the kernel, any access to the
relevant (virtual) address is not present on the GPU, and
the GPU will issue a page fault.

The relevant page of memory must then be migrated (i.e., copied)
from the host to the GPU before useful execution can continue.

Likewise, if the same data is required by the host after the kernel,
an access on the host will trigger a page fault on the CPU, and the
relevant data must be copied back from the GPU to the host.

### Prefetching

If the programmer knows in advance that memory is required on the
device before kernel execution, a prefetch to the destination
device may be issued. Schematically:

```cpp
  hipGetDevice(&device);
  hipMallocManaged(&ptr, nbytes);

  /* ... initialise data ... */

  hipMemPrefetchAsync(ptr, nbytes, device);

  /* ... kernel activity ... */
```

As the name suggests, this is an asynchronous call (it is likely to return
before any data transfer has actually occurred).
It can be viewed as a request to the HIP run-time to transfer the
data.

The memory must be managed by HIP.

Prefetches from the device to the host can be requested by using the special
destination value `hipCpuDeviceId`.

*Note:* Currently, the `hipMemPrefetchAsync` API is implemented on Linux and is
under development for Windows.

### Providing hints

Another mechanism to help the HIP run-time is to provide "advice".
This is done via

```cpp
  __host__ hipError_t hipMemAdvise(const void *ptr, size_t sz,
                                   hipMemoryAdvise advice, int device);
```

The `hipMemoryAdvise` value may include:

|  Advice |  Description |
|---|---|
|   `hipMemAdviseSetReadMostly` |  Data will mostly be read and only occasionally be written to |
|   `hipMemAdviseSetPreferredLocation` |  Set the preferred location for the data as the specified device |
| `hipMemAdviseSetAccessedBy` | Data will be accessed by the specified device so prevent page faults as much as possible |
| `hipMemAdviseSetCoarseGrain` | The default memory model is fine-grain. That allows coherent operations between host and device, while executing kernels. The coarse-grain can be used for data that only needs to be coherent at dispatch boundaries for better performance |

Each option has a corresponding `Unset` (e.g., `hipMemAdviseUnsetReadMostly`) 
value which can be used to nullify the effect of a preceding `Set`
specification. Further info can be found on
[the HIP Documentation.](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html)

Again, the relevant memory must be managed by HIP.

*Note:* Currently, the `hipMemAdvise` API and `hipMemoryAdvise` are implemented on
Linux and are under development for Windows.

## When to use

Often useful to start development and testing with managed memory, and
then move to explicit `hipMalloc()/hipMemcpy()` if it is required for
performance and is simple to do so.

## Exercise (15 minutes)

In the current directory we have supplied as a template the solution
to the exercise to the previous section. This just computes the
operation `A_ij := A_ij + alpha x_i y_j`.

It may be useful to run the unaltered code once to have a reference
`rocprof` output to show the times for different parts of the code.
`rocprof` is used in the submission script provided.

Confirm you can replace the explicit memory management using
`new/hipMalloc()` and `hipMemcpy()` with managed memory.
It is suggested that, e.g., both `d_a` and `h_a` are replaced
by the single declaration `a` in the main function.

Run the new code to check the answers are correct, and the new output
of `rocprof` associated with managed (unified) memory.

Add the relevant prefetch requests for the vectors `x` and `y` before
the kernel, and the matrix `a` after the kernel. Note that the device
id is already present in the code as `deviceNum`.

### What can go wrong?

What happens if you should accidentally use `hipMalloc()` where you intended
to use `hipMallocManaged()`?
