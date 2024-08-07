CC = hipcc
CXXFLAGS = -x hip -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH} -D__HIP_PLATFORM_AMD__ --offload-arch=gfx90a
SRC=$(wildcard *.cpp)

a.out: $(SRC)
	$(CC) $(CXXFLAGS) $^

clean:
	rm -f a.out
