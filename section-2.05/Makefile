CC = CC
CXXFLAGS = -x hip -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH}

all: exercise_ddot

exercise_ddot: exercise_ddot.hip.cpp
	$(CC) $(CXXFLAGS) exercise_ddot.hip.cpp

clean:
	rm -f a.out