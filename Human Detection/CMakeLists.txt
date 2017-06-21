cmake_minimum_required (VERSION 2.8)

project(detection)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{HOME}/freenect2/lib/cmake/freenect2)
find_package(freenect2 REQUIRED)

find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED core imgproc)

include_directories("/usr/include/libusb-1.0/")
include_directories(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(
  ${freenect2_INCLUDE_DIR}
)
add_executable(cv_detection test.cpp)

target_link_libraries(cv_detection ${OpenCV_LIBS})
target_link_libraries(cv_detection ${freenect2_LIBRARIES})
