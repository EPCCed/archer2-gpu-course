CC = CC
CXXFLAGS = -x hip -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH}

all: exercise_dscal

exercise_dscal: exercise_dscal.cpp
	$(CC) $(CXXFLAGS) exercise_dscal.cpp

clean:
	rm -f a.out