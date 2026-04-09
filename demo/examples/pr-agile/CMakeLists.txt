if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	cmake_minimum_required(VERSION 3.22)
	project(agile_demo_pr LANGUAGES CXX CUDA)

	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)

	find_package(CUDAToolkit REQUIRED)
	find_package(Boost REQUIRED)
	set(AGILE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
	set(DEMO_EXAMPLES_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
elseif(NOT DEFINED AGILE_ROOT)
	set(AGILE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_executable(agile_demo_pr
	src/main.cu
)

target_include_directories(agile_demo_pr PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${AGILE_ROOT}/include
	${AGILE_ROOT}/benchmarks/common
	${AGILE_ROOT}/common
	${DEMO_EXAMPLES_ROOT}/common
)

set_target_properties(agile_demo_pr PROPERTIES
	CUDA_ARCHITECTURES native
	CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_definitions(agile_demo_pr PRIVATE
	ENABLE_LOGGING=0
)

target_compile_options(agile_demo_pr PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -w>
)

if(TARGET Boost::headers)
	target_link_libraries(agile_demo_pr PRIVATE Boost::headers)
elseif(TARGET Boost::boost)
	target_link_libraries(agile_demo_pr PRIVATE Boost::boost)
endif()

target_link_libraries(agile_demo_pr PRIVATE
	CUDA::cudart
	CUDA::cuda_driver
	CUDA::cublas
	CLI11::CLI11
	rt
)
