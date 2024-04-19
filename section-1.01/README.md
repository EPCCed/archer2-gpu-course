# Introduction

- A simple model of performance
- Graphics processors

## A simple model of performance

A very simple picture of a computer might be

![A simple picture of CPU/memory](../images/ks-schematic-simple.svg)

Correspondingly, there might be a number of factors which could be taken into
consideration in a performance model:

1. **Clock speed:** the rate of issue of instructions by the processor
2. **Memory latency:** time taken to retrieve a data item from memory
3. **Memory bandwidth:** amount of data transferred in unit time
4. **Parallelism:** can I replicate the basic unit above?


### Clock speed

Processor clock speed determines the fundamental rate of processing
instructions, and hence data.

Historically, increases in CPU performance have been related to increases in
clock speed. However, owing largely to power constraints, most modern processors
have a clock speed of around 2-3 GHz.

Without some unforeseen fundamental breakthrough, it is not expected that this
fundamental speed will increase significantly in the future.

### Memory latency

Memory latency is a serious concern. It may take O(100-1000) clock cycles, from
a standing start, to retrieve a piece of data from memory to the processor
(where it can be held and operated on in a register).


CPUs mitigate this problem by having caches: memory that is "closer" to the
processor, thereby reducing access time. Many caches are hierarchical in nature:
the closer the processor, the smaller the cache size in bytes, but the faster
the access. These are typically referred to as Level 1, Level 2, Level 3, (L1,
L2, L3) and so on.

#### Exercise (1 minute)

Try the command
```bash
lscpu
```
on ARCHER2 to see what the cache hierarchy looks like.

Other latency hiding measures exist, e.g., out-of-order execution where
instructions are executed based on the availability of data, rather than the
order originally specified.

### Memory bandwidth

CPUs generally have commodity DRAM (dynamic random access memory). While the
overall size of memory can vary on O(100) GB, the exact size may be a cost
(money) decision. Similarly, the maximum memory bandwidth is usually limited to
O(100) GB/second for commodity hardware.

Note that many real applications are limited by memory bandwidth, where moving
data rather than performing arithmetic operations becomes the bottleneck.
Therefore, memory bandwidth is a crucial consideration.

### Parallelism

While it is not possible to increase the clock speed of an individual processor,
one can use/add more processing units (for which we will read: "cores").

Many CPUs are now multi-core or many-core, with perhaps O(10) or O(100) cores.
Applications wishing to take advantage of such architectures *must* be parallel.

#### Exercise (1 minute)

Look at `lscpu` again to check how many cores, or processors, are available on
ARCHER2.


## Graphics processors

Driven by commercial interest (games), a many-core processor *par excellence*
has been developed. These are graphics processors. Subject to the same
considerations as those discussed above, the hardware design choices taken to
resolve them have been specifically related to the parallel pixel rendering
problem (a trivially parallel problem).

Clocks speeds have, historically, lagged behind CPUs, but are now broadly
similar. However, increases in GPU performance are related to parallelism.

Memory latency has not gone away, but the mechanism used to mitigate it is to
allow very fast switching between parallel tasks. This eliminates idle time.

GPUs typically include high-bandwidth memory (HBM): a relatively small capacity
O(10) GB but relatively large bandwidth O(1000) GB/second configuration. This
gives a better balance between data supply and data processing for the parallel
rendering problem.

So GPUs have been specifically designed to solve the parallel problem of
rendering independent pixels. A modern GPU may have O(1000) cores.


### Hardware organisation

Cores on AMD GPUs are organised into units referred to as *compute unit*, or
CUs. Each CU contains several smaller processors called stream processors. These
processors are responsible for carrying out the mathematical operations that are
required for tasks such as rendering graphics, encoding video, and performing
scientific calculations.

For example, each AMD Instinct MI210 accelerator present in ARCHER2 boasts 104
compute units (CUs). Each CU houses 64 stream processors, resulting in a total
of 6,656 stream processors working in tandem.


A more complete overview is given by AMD:
[AMD Instinct Accelerators.](https://www.amd.com/en/technologies/cdna.html#instinct)

For NVIDIA GPUs, the picture is essentially similar, although some of the jargon
differs.


## Host/device picture

GPUs are typically 'hosted' by a standard CPU, which is responsible for
orchestration of GPU activities. In this context, the CPU and GPU are often
referred to as *host* and *device*, respectively.

![Host/device schematic](../images/ks-schematic-host-device.svg)


A modern configuration may see the host (a multi-core CPU) host 4-8 GPU devices.
