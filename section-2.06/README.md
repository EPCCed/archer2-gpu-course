# Constant memory

One further type of memory used in HIP programs is *constant* memory,
which is device memory to hold values which cannot be updated for
the duration of a kernel.

Physically, this is likely to be a small cache on each CU set aside for
the purpose.

This can provide fast (read-only) access to frequently used values.
It is a limited resource (exact capacity may depend on particular
hardware).

Key characteristics:
1. **Read-Only**: Constant memory is read-only for the GPU kernels. It is written to by the host (CPU) before the kernel execution.
2. **Cached**: It is cached on-chip, which means that access to constant memory is much faster than access to global memory.
3. **Limited size**: The size of constant memory is limited, typically around 64KB, depending on the GPU architecture.

Benefits of using constant memory:
1. **Broadcasting**: When all threads in a warp access the same address in constant memory, the value is broadcast to all threads, resulting in a single memory read.
2. **Reduced latency**: Accessing constant memory is faster than accessing global memory due to its caching mechanism.
3. **Efficiency**: It is ideal for storing constants that are used by all threads, such as coefficients in mathematical formulas.

Common use cases:
1. **Lookup tables**: Storing lookup tables that are frequently accessed by the kernel.
2. **Constants**: Storing constants that are used across many threads, such as physical constants or coefficients.
3. **Configuration data**: Storing configuration parameters that do not change during kernel execution.

## Kernel parameters

If one calls a kernel function, actual arguments are (conceptually, at
least) passed by value as in standard C++, and are placed in constant memory.
E.g.,
```c
__global__ void kernel(double arg1, double *arg2, ...);
```
<!-- If one uses the `--ptxas-options=-v` option to `nvcc` this will
report (amongst other things) a number of `cmem[]` entries;
`cmem[0]` will usually include the kernel arguments. -->

Note this may be of limited size (e.g., 4096 bytes), so large
objects should not be passed by value to the device.

### Static

It is also possible to use the `__constant__` memory space qualifier
for objects declared at file scope, e.g.:
```c
static __constant__ double data_read_only[3];
```
Host values can be copied to the device with the API function
```c
  double values[3] = {1.0, 2.0, 3.0};

  hipMemcpyToSymbol(data_read_only, values, 3*sizeof(double));
```
The object `data_read_only` may then be accessed by a kernel or kernels
at the same scope.

<!-- The compiler usually reports usage under `cmem[3]`. -->
Again, capacity is limited (e.g., may be 64 kB). If an object is too large it
will probably spill into global memory.

## Exercise

We should now be in a position to combine our matrix operation, and
the reduction required for the vector product to perform another
useful operation: a matrix-vector product. For a matrix `A_mn` of
`m` rows and `n` columns, the product `y_i = alpha A_ij x_j` may be
formed with a vector `x` of length `n` to give a result `y` of
length `m`. `alpha` is a constant.

A new template has been provided. A simple serial version has been
implemented with some test values.

Suggested procedure:
1. To start, make the simplifying assumption that we have only 1 block
   per row, and that the number of columns is equal to the number of
   threads per block. This should allow the elimination of the loop
   over both rows and columns with judicious choice of thread indices.
2. The limitation to one block per row may harm occupancy. So we
   need to generalise to allow columns to be distributed between
   different blocks. Hint: you will probably need a two-dimensional
   `__shared__` provision in the kernel. Use the same total number
   of threads per block with `blockDim.x == blockDim.y`.

   Make sure you can increase the problem size (specifically, the number
   of columns `ncol`) and retain the correct answer.
3. Leave the concern of coalescing until last. The indexing can be rather
   confusing. Again, remember to deal with any array 'tails'.

### Finished?

A fully robust solution might check the result with a rectangular
thread block.

A level 2 BLAS implementation may want to compute the update
` y_i := alpha A_ij x_j + beta y_i`. How does this complicate
the simple matrix-vector update?
