
# The HIP/CUDA programming model

The application programmer does not want to have to worry about
the exact disposition of cores/CUs, or whatever, in the hardware.
An abstraction is wanted.

In HIP and CUDA, this abstraction is based on an hierarchical
organisation of threads.


## Decomposition into threads

If we have a one-dimensional problem, e.g., an array, we can assign individual
elements to threads.

![A single thread block in one dimension](../images/ks-threads.jpeg)

### Workgroups

A *workgroup* (block) is a collection of work-items (or threads) scheduled to
run on a single Compute Unit (CU). In our one-dimensional picture we may have:

![Threads and blocks in one dimension](../images/ks-threads-blocks.jpeg)

A *wavefront*, a subset of a work group, is designated to execute as a single
SIMD (Single Instruction Multiple Data) unit. In AMD GPUs, wavefronts commonly consist of 64 threads.

Wavefronts are mapped to Compute Units (CU) for execution by the hardware
scheduler.

Developers typically select the number of wavefronts/workgroup, often based on
the problem's size.


### Two dimensions

For two-dimensional problems (e.g., images) it is natural to have a
two-dimensional Cartesian picture:

![Threads and blocks in two dimensions](../images/ks-threads-blocks-grids.jpeg)

The arrangement of workgroups (blocks) is referred to as the *grid*.

CUDA and HIP allow the picture to be extended straightforwardly
to three dimensions.


## Programming

HIP, developed by AMD, stands for Heterogeneous-compute Interface for
Portability. It's a runtime API and kernel language implemented in C++. This
tool enables developers to create applications that are portable across AMD's
accelerators and CUDA devices alike.

HIP has two distinct types of source code: Host code and Device code. The Host,
essentially the CPU, executes the Host code. This code has a typical C++ syntax
and incorporates standard features. The entry point for execution is the 'main'
function. Utilizing the HIP API within, Host code facilitates tasks such as
generating device buffers, managing data transfer between the host and device,
and initiating the execution of device code.

The Device, typically the GPU, executes the Device code. It utilizes a C-like
syntax, where kernels are used to launch device code. Instructions from the
Host are queued into streams for execution on the device.

## Compilation

`hipcc` is a compiler driver utility that calls the AMD LLVM compiler
`amdclang(++)` (or `nvcc`) to compile HIP code. `hipcc` is included in the AMD
ROCm Software stack.

`hipcc` compiles both HIP code for GPU execution and non-HIP code for CPU
execution, defaulting to AMD LLVM compiler `amdclang(++)`. Other compilers can
be used for non-HIP code, and object files can be linked accordingly.

Use `hipcc --help` for a list of options. To compile code, use: Eg.
```
$ hipcc -x hip code.cpp
```
where the `-x hip` option instructs `hipcc` to interpret code as HIP specific.

To see what `hipcc` passes to the compiler, you can pass the `--verbose` option.

### Compute capabilities

Different generations of hardware have different capabilities in terms
of the features they support.

The consequence is that a program must be compiled for the relevant
architecture to be able to run. E.g.,
```
$ hipcc --offload-arch=gfx90a -x hip code.cpp
```
will run  on a subset of AMD Radeon Instinct series of accelerators

The `--offload-arch=<value>` option enables CUDA offloading device architecture,
or HIP offloading target ID in the form of a device architecture. The available
values for AMD accelerators can be found in
https://llvm.org/docs/AMDGPUUsage.html#processors.

## Portability: CUDA and HIP

CUDA has been under development by NVIDIA since around 2005. AMD, rather
later to the party, develops HIP, which shadows CUDA. For
example, a C/C++ call to
```
  cudaMalloc(...);
```
is simply replaced by
```
  hipMalloc(...);
```
with the same signature. HIP code can be compiled for NVIDIA GPUs by
inclusion of an appropriate wrapper which just substitutes the relevant
CUDA API routine.

Not all the latest CUDA functionality is implemented in HIP at any given
time.

## Summary

The goal for the programmer is to describe the problem in the
abstraction of grids, blocks, threads. The hardware is then
free to schedule work as it sees fit.

This is the basis of the scalable parallelism of the architecture.

The very latest HIP programming guide is well worth a look.

https://rocm.docs.amd.com/projects/HIP/en/latest/
