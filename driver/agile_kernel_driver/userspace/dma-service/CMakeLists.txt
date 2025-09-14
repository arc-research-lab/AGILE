cmake_minimum_required(VERSION 3.22)
project(cuda_hello LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU archs (no -arch in flags)
set(CMAKE_CUDA_ARCHITECTURES 80 86)

# Policy: prefer BoostConfig
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# Dependencies
# Tell CMake to use Boost's package config (requires Boost_DIR or CMAKE_PREFIX_PATH)
find_package(Boost 1.89.0 CONFIG REQUIRED COMPONENTS program_options)
find_package(CUDAToolkit REQUIRED)

add_executable(app
  src/main.cu
)

# Includes
target_include_directories(app PRIVATE
  /home/zhuoping/workspace/linux-kernel/linux-kernel/agile_kernel_driver/common
  /home/zhuoping/workspace/AGILE/driver/gdrcopy/include
  /home/zhuoping/workspace/linux-kernel/linux-kernel/agile_gpu/common
)

# Compile options
target_compile_options(app PRIVATE -O3 -w)
set_target_properties(app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(app PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unknown-pragmas>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC,-fno-omit-frame-pointer>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>   # (drop -arch=native)
)

# (Optional) static libstdc++/libgcc — remove if it causes device-link issues
# target_link_options(app PRIVATE -static-libstdc++ -static-libgcc)

# RPATH so the runtime finds your GCC14 + local Boost
target_link_options(app PRIVATE
  "-Wl,-rpath,$ENV{HOME}/.local/gcc-14.3.0/lib64:$ENV{HOME}/.local/boost-1.89/lib"
)

# Link with imported CUDA targets + Boost
target_link_libraries(app PRIVATE
  CUDA::cudart
  CUDA::cublas          # was 'cublas'
  CUDA::cuda_driver     # was 'cuda' (driver API)
  pthread dl rt numa
  Boost::program_options
  Boost::headers
)

# Extra RPATHs (binary-dir relative + gdrcopy location)
set_target_properties(app PROPERTIES
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/lib"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/lib;"
  POSITION_INDEPENDENT_CODE ON
)
