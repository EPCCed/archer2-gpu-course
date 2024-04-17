COMPILER = CC

COMPILER_FLAGS = -x hip -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH}

# Target exercise/file
TARGET = exercise_dger

SRC = $(TARGET).hip.cpp

# All rule (no changes)
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(COMPILER) $(COMPILER_FLAGS) $(SRC)

# Clean rule (no changes)
clean:
	rm -f a.out