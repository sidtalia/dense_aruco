cmake_minimum_required(VERSION 3.18)
project(dense_aruco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Do not require objdetect: OpenCV < 4.7 keeps ArUco in opencv_contrib (aruco.hpp).
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
set(DENSE_ARUCO_OPENCV_LIBS ${OpenCV_LIBS})
foreach(_comp objdetect aruco)
  find_package(OpenCV QUIET COMPONENTS ${_comp})
  if(TARGET opencv_${_comp})
    list(APPEND DENSE_ARUCO_OPENCV_LIBS opencv_${_comp})
  endif()
endforeach()

find_package(Threads REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
  src/bindings.cpp
  src/detector.cpp
  src/refine.cpp
)
target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(_core PRIVATE ${DENSE_ARUCO_OPENCV_LIBS} Threads::Threads)
target_compile_options(_core PRIVATE -O3 -DNDEBUG)

install(TARGETS _core DESTINATION dense_aruco)
