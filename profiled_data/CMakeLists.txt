cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(profiler)

set(CMAKE_PREFIX_PATH "/usr/local/include/libtorch/share/cmake/Torch")
# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})
set(TensorRT_DIR /usr/src/tensorrt/)

find_package(TensorRT REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost COMPONENTS thread system REQUIRED)
find_package (OpenCV  REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
message(${Boost_INCLUDE_DIR})
include_directories(${TensorRT_INCLUDE_DIRS} ${TensorRT_DIR}/samples/common ${CUDA_INCLUDE_DIRS})
include_directories (${OpenCV_INCLUDE_DIRS})

include_directories(/home/ncl/ktd/prj/thesis/efficient-serving-system/include)

add_executable(cpp_profiler tcpp_profiler.cpp)
target_link_libraries(cpp_profiler ${TORCH_LIBRARIES} rt)

add_executable(trt_profiler trt_profiler.cpp /home/ncl/ktd/prj/thesis/efficient-serving-system/src/TRTengine.cpp)
target_link_libraries(trt_profiler rt ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} )

add_executable(main main.cpp)

set_property(TARGET main cpp_profiler PROPERTY CXX_STANDARD 17)