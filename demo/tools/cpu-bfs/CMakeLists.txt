if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	cmake_minimum_required(VERSION 3.22)
	project(agile_cpu_bfs LANGUAGES CXX)

	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_executable(agile_cpu_bfs
	main.cc
)

target_compile_options(agile_cpu_bfs PRIVATE -O3 -Wall)

target_link_libraries(agile_cpu_bfs PRIVATE CLI11::CLI11)
