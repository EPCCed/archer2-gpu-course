COMPILER = CC

COMPILER_FLAGS = -g -x hip -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH} -I${CRAY_MPICH_DIR}/include -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a}

# Target exercise/file
TARGET = test

SRC = $(TARGET).hip.cpp

# All rule (no changes)
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(COMPILER) $(COMPILER_FLAGS) $(SRC)

# Clean rule (no changes)
clean:
	rm -f a.out