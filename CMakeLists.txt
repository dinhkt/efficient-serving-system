cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(app)


set(CMAKE_PREFIX_PATH "/home/ncl/ktd/Download/libtorch/share/cmake/Torch;/usr/lib/debug/lib/x86_64-linux-gnu/")
# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(TensorRT_DIR /usr/src/tensorrt/)

find_package(TensorRT REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost COMPONENTS thread system REQUIRED)
find_package (OpenCV  REQUIRED)

find_package(CUDA REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${TensorRT_INCLUDE_DIRS} ${TensorRT_DIR}/samples/common ${CUDA_INCLUDE_DIRS})
include_directories (${OpenCV_INCLUDE_DIRS})

add_executable(infer infer.cpp include/imageutils.cc)
target_link_libraries(infer "${TORCH_LIBRARIES}" ${OpenCV_LIBS} rt)

add_executable(infer_trt infer_trt.cpp include/TRTengine.cc)
target_link_libraries(infer_trt ${OpenCV_LIBS} rt ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} )

add_executable(server server.cpp include/IPManager.cc include/base64.cc include/imageutils.cc)
target_link_libraries(server "${TORCH_LIBRARIES}" rt ${Boost_LIBRARIES} ${OpenCV_LIBS} )

set_property(TARGET server infer infer_trt PROPERTY CXX_STANDARD 14)
