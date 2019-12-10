cmake_minimum_required(VERSION 3.10)

set(LIBRARY_NAME SolveHomography)
project(${LIBRARY_NAME})

set(CMAKE_CXX_STANDARD 11)

#添加opencv库
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# 搜集源码文件
file(GLOB SRC ../*.cpp)

set(SOURCE_FILES ${SRC})

add_executable(${LIBRARY_NAME} ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} ${OpenCV_LIBS})
