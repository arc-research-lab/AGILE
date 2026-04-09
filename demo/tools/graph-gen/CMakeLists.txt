if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	cmake_minimum_required(VERSION 3.22)
	project(agile_graph_gen LANGUAGES CXX)

	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# gapbs source location
set(GAPBS_SRC_DIR "/home/zhuoping/workspace/gapbs/src" CACHE PATH "Path to gapbs src/ directory")

find_package(OpenMP REQUIRED)

add_executable(agile_graph_gen
	main.cc
)

target_include_directories(agile_graph_gen PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${GAPBS_SRC_DIR}
)

target_compile_options(agile_graph_gen PRIVATE -O3 -Wall)

target_link_libraries(agile_graph_gen PRIVATE OpenMP::OpenMP_CXX)
