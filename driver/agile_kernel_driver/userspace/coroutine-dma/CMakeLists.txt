cmake_minimum_required(VERSION 3.22)
project(cuda_hello LANGUAGES CXX CUDA)

# Host/CUDA language standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architectures
set(CMAKE_CUDA_ARCHITECTURES 80;86)


# CUDA toolkit
find_package(CUDAToolkit REQUIRED)

add_executable(app
  src/main.cu
  src/common.cpp
)

# Includes
target_include_directories(app PRIVATE
  /home/zhuoping/workspace/linux-kernel/linux-kernel/agile_kernel_driver/common
  /home/zhuoping/workspace/AGILE/driver/gdrcopy/include
)
target_compile_options(app PRIVATE -w)
# If you need device code split across multiple .cu files:
set_target_properties(app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Compile options
target_compile_options(app PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unknown-pragmas>
  # nvcc host flags (avoid duplicating -std=c++20 since we set standards above)
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC,-fno-omit-frame-pointer>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>: -w -Xptxas -v -arch=native>
)

# GDRCopy shared library
set(GDRCOPY_LIB /home/zhuoping/workspace/AGILE/driver/gdrcopy/src/libgdrapi.so)
get_filename_component(GDRCOPY_DIR "${GDRCOPY_LIB}" DIRECTORY)

# Link CUDA runtime + cublas + gdrcopy + common system libs
target_link_libraries(app PRIVATE
  CUDA::cudart
  cublas
  cuda
  "${GDRCOPY_LIB}"
  pthread
  dl
  rt
)

# Statically link libstdc++ and libgcc (keeps glibc dynamic)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(app PRIVATE -static-libstdc++ -static-libgcc)
endif()

# RPATH so the binary can find gdrcopy at runtime (and any adjacent libs)
# $ORIGIN makes it relative to the binary location.
set_target_properties(app PROPERTIES
  BUILD_RPATH "${GDRCOPY_DIR};\$ORIGIN;\$ORIGIN/lib"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/lib;${GDRCOPY_DIR}"
  POSITION_INDEPENDENT_CODE ON
)
