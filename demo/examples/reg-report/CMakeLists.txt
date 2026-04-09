if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	cmake_minimum_required(VERSION 3.22)
	project(agile_demo_reg_report LANGUAGES CXX)

	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)

	find_package(Boost REQUIRED)
	set(AGILE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
	set(DEMO_EXAMPLES_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
elseif(NOT DEFINED AGILE_ROOT)
	set(AGILE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_executable(agile_demo_reg_report
	src/main.cpp
)

target_include_directories(agile_demo_reg_report PRIVATE
	${DEMO_EXAMPLES_ROOT}/common
)

target_compile_features(agile_demo_reg_report PRIVATE cxx_std_17)

if(TARGET Boost::headers)
	target_link_libraries(agile_demo_reg_report PRIVATE Boost::headers)
elseif(TARGET Boost::boost)
	target_link_libraries(agile_demo_reg_report PRIVATE Boost::boost)
endif()

target_link_libraries(agile_demo_reg_report PRIVATE
	CLI11::CLI11
	rt
	pthread
)
