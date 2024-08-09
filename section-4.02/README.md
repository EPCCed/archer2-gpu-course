# Graph API

Streams provide a mechanism to control the execution of independent,
or asynchronous, work.

A more general mechanism, added more recently in HIP, introduces the
idea of a graph.

Graphs can be used to orchestrate complex workflows, and may be
particularly useful in amortising the overhead of many small
kernel launches.

<!-- Note: the latest HIP does support a subset of Graph API operations,
but I haven't had a chance to try it out yet. -->

<!-- This graph APIs are marked as beta, meaning, while this is feature complete,
it is still open to changes and may have outstanding issues. -->


## Graphs

The idea (taken from graph theory) is to represent individual executable
items of work by the nodes of the graph, and the dependencies between
them as edges connecting the relevant nodes.

![Some simple graph structures](../images/graph.svg)

There is the assumption that there is a beginning and an end (ie., there
is a direction), and that there are no closed loops in the graph picture.
This gives rise to a *Directed Acyclic Graph*, or DAG.

The idea is then to construct a description of the graph from the
constituent nodes and dependencies, and then execute the graph.

### Creating a HIP graph

The overall container for a graph is of type
```cpp
  hipGraph_t graph;
```
and is allocated in an empty state via the API function
```cpp
  __host__ hipError_t hipGraphCreate(hipGraph_t *graph, unsigned int flags);
```

The only valid value for the second argument is `flags = 0`.

For example, the life-cycle would typically be:
```cpp
  hipGraph_t myGraph;

  hipGraphCreate(&myGraph, 0);

  /* ... work ... */

  hipGraphDestroy(myGraph);
```
Destroying the graph will also release any component nodes/dependencies.


### Instantiating and executing a graph

Having created a graph object, one needs to add nodes and dependencies
to it. When this has been done (adding nodes etc will be discussed below),
one creates an executable graph object
```cpp
  hipGraphExec_t graphExec;

  hipGraphInstantiate(&graphExec, myGraph, NULL, NULL, 0);

  /* ... and launch the graph into a stream ... */

  hipGraphLaunch(graphExec, stream);

  hipGraphExecDestroy(graphExec);
```
The idea here is that the instantiation step performs a lot of the
overhead of setting up the launch parameters and so forth, and then
the launch is relatively small compared with a standard launch.


## Graph definition

The following sections consider the explicit definition of graph
structure.

### Node types

The nodes of the graph may represent a number of different types of
operation. Valid choices include:

1. A `hipMemcpy()` operation
2. A `hipMemset()` operation
3. A kernel
4. A CPU function call

Specifying the nodes of a graph means providing a description of the
arguments which would have been used in a normal invocation, such as
those we have seen for `hipMemcpy()` before.

### Kernel node

Suppose we have a kernel function with arguments
```cpp
  __global__ void myKernel(double a, double * x);
```
and which is executed with configuration including `blocks` and
`threadsPerBlock`.

These parameters are described in HIP by a structure `hipKernelNodeParams`
which includes the public members:
```cpp
   void *func;             /* pointer to the kernel function */
   void **kernelParams;    /* List of kernel arguments */
   dim3 gridDim;           /* Number of blocks */
   dim3 blockDim;          /* Number of threads per block */
```
So, with the relevant host variables in scope, we might write
```cpp
  hipKernelNodeParams kParams = {0};    /* Initialise to zero */
  void * args[] = {&a, &d_x};           /* Kernel arguments */

  kParams.func         = (void *)myKernel;
  kParams.kernelParams = args;
  kParams.gridDim      = blocks;
  kParams.blockDim     = threadsPerBlock;
```
We are now ready to add a kernel node to the graph (assumed to
be `myGraph`):
```cpp
  hipGraphNode_t kNode;     /* handle to the new kernel node */

  hipGraphAddKernelNode(&kNode, myGraph, NULL, 0, &kParams);
```
This creates a new kernel node, adds it to the existing graph, and
returns a handle to the new node.

The formal description is
```cpp
__host__ hipError_t hipGraphAddKernelNode(hipGraphNode_t *node,
                                          hipGraph_t graph,
					                                const hipGraphNode_t *dependencies,
					                                size_t nDependencies,
					                                const hipKernelNodeParams  *params);
```
If the new node is not dependent on any other node, then the third and
fourth arguments can be `NULL` and zero, respectively.

### A `memcpy` node

There is a similar procedure to define a `memcpy` node. We need the
structure `hipMemcpy3DParms` (sic) with relevant public members
```cpp
  struct hipPos          srcPos;       /* offset in source */
  struct hipPitchedPtr   srcPtr;       /* address and length in source */
  struct hipPos          dstPos;       /* offset in destination */
  struct hipPitchedPtr   dstPtr;       /* address and length in destination */
  struct hipExtent       extent;       /* dimensions of block */
  enum hipMemcpykind       kind;         /* direction of the copy */

```
This is rather involved, as it must allow for the most general
type of copy allowed in the HIP API.

Further details can be found on [hipMemcpy3DParms Struct Reference.](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_memcpy3_d_parms.html)

To make this more concrete, consider an explicit `hipMemcpy()` operation
```cpp
  hipMemcpy(d_ptr, h_ptr, ndata*sizeof(double), hipMemcpyHostToDevice);
```
We should then define something of the form
```cpp
  hipGraphNode_t node;
  hipMemcpy3DParms mParams = {0};

  mParams.kind   = hipMemcpyHostToDevice;
  mParams.extent = make_hipExtent(ndata*sizeof(double), 1, 1);
  mParams.srcPos = make_hipPos(0, 0, 0);
  mParams.srcPtr = make_hipPitchedPtr(h_ptr, ndata*sizeof(double), ndata, 1);
  mParams.dstPos = make_hipPos(0, 0, 0);
  mParams.dstPtr = make_hipPitchedPtr(d_ptr, ndata*sizeof(double), ndata, 1);
```
For simple one-dimensional allocations, it is possible to write some
simple helper functions to hide this complexity.

The information is added via:
```cpp
  hipGraphAddMemcpyNode(&mNode, myGraph, &kNode, 1, &mParams);
```
where we have made it dependent on the preceding kernel node.


These data structures are documented more fully in the data structures
section of the hip runtime API.

https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/annotated.html

## Synchronisation

If executing a graph in a particular stream, one can use
```cpp
  hipStreamSynchronize(stream);
```
to ensure that the graph is complete. It is also possible to use
```cpp
  hipDeviceSynchronize();
```
which actually synchronises all streams running on the current
device.


## Exercise (30 minutes)

The exercise revisits again the problem for `A_ij := A_ij + x_i y_j`,
and the exercise is to see whether you can replace the single
kernel launch with the execution of a graph. When you have a
working program, check with `rocprof` that this is doing what
you expect.

A new template is supplied if you wish to start afresh.

While it will not be possible to see any performance improvement
associated with this single kernel launch, the principle should
be clear.

### Finished?

Have a go at adding to the graph the dependent operation which is the
device to host `hipMemcpy()` of the result.
