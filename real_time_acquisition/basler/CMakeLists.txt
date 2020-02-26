cmake_minimum_required(VERSION 2.8.3)
project(DepthCapture)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../../cmake/Modules)

find_package(BaslerToF REQUIRED)
#find_package(OpenCV REQUIRED)

add_definitions(${BaslerToF_DEFINITIONS})
include_directories(${BaslerToF_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/DepthCapture.cpp)
target_link_libraries(${PROJECT_NAME} ${BaslerToF_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_FLAGS ${BaslerToF_LINKER_FLAGS})
