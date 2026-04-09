if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	cmake_minimum_required(VERSION 3.22)
	project(agile_demo_pr_bam LANGUAGES CXX CUDA)

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

# ── BAM (libnvm) paths ──
set(BAM_ROOT ${AGILE_ROOT}/baseline/bam CACHE PATH "Path to BAM source root")
set(BAM_LIB_DIR ${BAM_ROOT}/build/lib CACHE PATH "Path to built libnvm.so")

if(NOT EXISTS ${BAM_ROOT}/include/page_cache.h)
	message(WARNING "BAM headers not found at ${BAM_ROOT}/include — skipping pr-bam")
	return()
endif()
if(NOT EXISTS ${BAM_LIB_DIR}/libnvm.so)
	message(WARNING "libnvm.so not found at ${BAM_LIB_DIR} — skipping pr-bam (build BAM first)")
	return()
endif()

add_executable(agile_demo_pr_bam
	src/main.cu
)

target_include_directories(agile_demo_pr_bam PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${AGILE_ROOT}/include
	${AGILE_ROOT}/benchmarks/common
	${AGILE_ROOT}/common
	${DEMO_EXAMPLES_ROOT}/common
	${BAM_ROOT}/include
	${BAM_ROOT}/include/freestanding/include
)

set_target_properties(agile_demo_pr_bam PROPERTIES
	CUDA_ARCHITECTURES native
	CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_definitions(agile_demo_pr_bam PRIVATE
	ENABLE_LOGGING=0
)

target_compile_options(agile_demo_pr_bam PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -w -Xcompiler=-fpermissive>
)

if(TARGET Boost::headers)
	target_link_libraries(agile_demo_pr_bam PRIVATE Boost::headers)
elseif(TARGET Boost::boost)
	target_link_libraries(agile_demo_pr_bam PRIVATE Boost::boost)
endif()

target_link_libraries(agile_demo_pr_bam PRIVATE
	CUDA::cudart
	CUDA::cuda_driver
	CLI11::CLI11
	${BAM_LIB_DIR}/libnvm.so
	rt
)
