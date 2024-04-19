CC = CC
CXXFLAGS = -x hip -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH}

all: exercise_dscal

exercise_dscal: exercise_dscal.hip.cpp
	$(CC) $(CXXFLAGS) exercise_dscal.hip.cpp

clean:
	rm -f a.out
