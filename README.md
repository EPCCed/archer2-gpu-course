
<img src="./images/archer2_logo.png" align="left" width="284" height="80" />
<img src="./images/epcc_logo.png" align="right" width="248" height="66" />

<br /><br /><br /><br />

# Introduction to GPU programming with HIP

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This short course will provide an introduction to GPU computing with HIP
aimed at scientific application programmers wishing to develop their own
software. The course will give a background on the difference between CPU
and GPU architectures as a prelude to introductory exercises in HIP
programming. The course will discuss the execution of kernels, memory
management, and shared memory operations. Common performance issues are
discussed and their solution addressed.

<!-- Profiling will be introduced via
the current AMD tools. -->

The course will go on to consider execution of independent streams, and
the execution of work composed as a collection of dependent tasks expressed
as a graph. Device management and details of device to device data transfer
will be covered for situations where more than one GPU device is available.
<!-- HIP-aware MPI will be covered. -->

The course will not discuss programming with compiler directives, but does
provide a concrete basis of understanding of the underlying principles of
the HIP model which is useful for programmers ultimately wishing to make
use of OpenMP or OpenACC (or indeed other models). The course will not
consider graphics programming, nor will it consider machine learning
packages.

Note that the course is also appropriate for those wishing to use NVIDIA GPUs
via the CUDA API, although we will not specifically use CUDA.

Attendees must be able to program in C or C++ (course examples and
exercises will limit themselves to C++). A familiarity with threaded
programming models would be useful, but no previous knowledge of GPU
programming is required.

## Installation

For details of how to log into an ARCHER2 account, see the
[ARCHER2 quickstart for users.](https://docs.archer2.ac.uk/quick-start/quickstart-users/)

Check out the Git repository to your ARCHER2 account.
```bash
cd ${HOME/home/work}
git clone https://github.com/EPCCed/archer2-gpu-course.git
cd archer2-gpu-course
```

For the examples and exercises in the course, we will use the
AMD compiler driver. To access this
```bash
module load PrgEnv-amd
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan
```

Check you can compile and run a very simple program
and submit the associated script to the queue system.
```bash
cd section-2.01
hipcc -x hip -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH}  -D__HIP_PLATFORM_AMD__ --offload-arch=gfx90a exercise_dscal.hip.cpp
sbatch submit.sh
```

The result should appear in a file `slurm-123456.out` in the working
directory.

Each section of the course is associated with a different directory, each
of which contains a number of example programs and exercise templates.
Answers to exercises generally re-appear as templates to later exercises.
Miscellaneous solutions also appear in the solutions directory.


## Timetable

The timetable may shift slightly in terms of content, but we will stick to
the advertised start and finish times, and the break times.


### Day one

| Time  | Content                                  | Section                      |
|-------|------------------------------------------|------------------------------|
| 10:00 | Logistics, login, modules, local details | See above                    |
| 10:15 | Introduction                             |                              |
|       | Performance model; Graphics processors   | [section-1.01](section-1.01) |
| 10:45 | Morning break                            |                              |
| 11:00 | The CUDA/HIP programming model           |                              |
|       | Abstraction; host code and device code   | [section-1.02](section-1.02) |
| 12:00 | Lunch                                    |                              |
| 13:00 | CUDA/HIP programming: memory management  |                              |
|       | `hipMalloc(), hipMemcpy()`               | [section-2.01](section-2.01) |
| 13:45 | Executing a kernel                       |                              |
|       | `__global__` functions `<<<...>>>`       | [section-2.02](section-2.02) |
| 14:30 | Afternoon break                          |                              |
| 14:45 | Some performance considerations          |                              |
|       | Exercise on matrix operation             | [section-2.03](section-2.03) |
| 16:00 | Close                                    |                              |

### Day two

| Time  | Content                                  | Section                      |
|-------|------------------------------------------|------------------------------|
| 10:00 | Review previous day briefly and solution of previous exercise |         |
| 10:45 | Morning break                            |                              |
| 11:00 | Managed memory                           |                              |
|       | Exercise on managed memory               | [section-2.04](section-2.04) |
| 12:00 | Lunch                                    |                              |
| 13:00 | Shared memory                            |                              |
|       | Exercise on vector product               | [section-2.05](section-2.05) |
| 13:45 | Constant memory                          |                              |
|       | All together: matrix-vector product      | [section-2.06](section-2.06) |
| 14:30 | Afternoon break                          |                              |
| 14:45 | Streams                                  |                              |
|       | Using `hipMempcyAsync()` etc             | [section-4.01](section-4.01) |
| 16:00 | Close                                    |                              |

### Day three


| Time  | Content                                  | Section                      |
|-------|------------------------------------------|------------------------------|
| 10:00 | Review previous day briefly and solution of previous exercise |         |
| 10:45 | Morning break                            |                              |
| 11:00 | Graph API                                |                              |
|       | Using `hipGraphLaunch()` etc             | [section-4.02](section-4.02) |
| 12:00 | Lunch                                    |                              |
| 13:00 | Device management: more then one GPU     |                              |
|       | `hipMemcpy()` again                      | [section-5.01](section-5.01) |
| 14:00 | Afternoon break                          |                              |
| 14:15 | Putting it all together	               |                              |
|       | Conjugate gradient exercise              | [section-6.01](section-6.01) |
| 16:00 | Close                                    |                              |


---
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]
