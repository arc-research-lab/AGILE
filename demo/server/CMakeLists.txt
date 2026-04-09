add_executable(websocket_demo_server main.cpp)

# include agile/include and demo common headers (demo_ipc.h, demo_examples.h)
target_include_directories(websocket_demo_server PRIVATE
	${CMAKE_SOURCE_DIR}/../include
	${DEMO_EXAMPLES_ROOT}/common
)

target_compile_options(websocket_demo_server PRIVATE -Wall -Wextra -Wpedantic)


if(TARGET Boost::headers)
	target_link_libraries(websocket_demo_server PRIVATE Boost::headers CLI11::CLI11 Threads::Threads rt)
elseif(TARGET Boost::boost)
	target_link_libraries(websocket_demo_server PRIVATE Boost::boost CLI11::CLI11 Threads::Threads rt)
else()
	target_link_libraries(websocket_demo_server PRIVATE CLI11::CLI11 Threads::Threads rt)
endif()