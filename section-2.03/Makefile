CC = CC
CXXFLAGS = -x hip -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH}

all: exercise_dger

exercise_dger: exercise_dger.hip.cpp
	$(CC) $(CXXFLAGS) exercise_dger.hip.cpp

clean:
	rm -f a.out