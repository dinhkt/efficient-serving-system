cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(app)

set(CMAKE_PREFIX_PATH "/home/ncl/ktd/Download/libtorch/share/cmake/Torch;/usr/lib/debug/lib/x86_64-linux-gnu/")
find_package(Torch REQUIRED)
find_package(Boost COMPONENTS thread system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
find_package (OpenCV  REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

add_executable(infer infer.cpp include/base64.cc include/imageutils.cc )
target_link_libraries(infer "${TORCH_LIBRARIES}" ${OpenCV_LIBS} rt)

add_executable(server server.cpp include/IPManager.cc)
target_link_libraries(server rt ${Boost_LIBRARIES} )

set_property(TARGET server PROPERTY CXX_STANDARD 14)
