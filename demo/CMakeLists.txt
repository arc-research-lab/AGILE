cmake_minimum_required(VERSION 3.22)

project(agile_demo_examples LANGUAGES CXX CUDA)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(AGILE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. CACHE INTERNAL "AGILE repository root")
set(DEMO_EXAMPLES_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/examples CACHE INTERNAL "Demo examples root")

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

FetchContent_Declare(
	CLI11
	GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
	GIT_TAG v2.4.2
	GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(CLI11)

file(GLOB example_entries RELATIVE ${DEMO_EXAMPLES_ROOT} ${DEMO_EXAMPLES_ROOT}/*)


foreach(example_dir IN LISTS example_entries)
	if(EXISTS ${DEMO_EXAMPLES_ROOT}/${example_dir}/CMakeLists.txt)
		add_subdirectory(${DEMO_EXAMPLES_ROOT}/${example_dir})
	endif()
endforeach()

# ── Tools ──
set(DEMO_TOOLS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tools CACHE INTERNAL "Demo tools root")
file(GLOB tool_entries RELATIVE ${DEMO_TOOLS_ROOT} ${DEMO_TOOLS_ROOT}/*)

foreach(tool_dir IN LISTS tool_entries)
	if(EXISTS ${DEMO_TOOLS_ROOT}/${tool_dir}/CMakeLists.txt)
		add_subdirectory(${DEMO_TOOLS_ROOT}/${tool_dir})
	endif()
endforeach()

# -- Server --
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/server)