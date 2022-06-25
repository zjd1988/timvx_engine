cmake_minimum_required(VERSION 3.10.1)
project(timvx_engine)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ")

set(TIM_VX_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/src/3rd_party/timvx/include)
set(JSON_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/src/3rd_party/nlohmann-json/include)
option(BUILD_SHARED_LIBS "build shared or static lib" ON)
file(GLOB_RECURSE SRC  ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if (BUILD_SHARED_LIBS)
    add_library(timvx_engine SHARED ${SRC})
else()
    add_library(timvx_engine STATIC ${SRC})
endif()

target_include_directories(timvx_engine PUBLIC 
    ${TIM_VX_INCLUDE_PATH}
    ${JSON_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(timvx_engine PUBLIC 
    ${PROJECT_SOURCE_DIR}/../../host_install/lib/libtim-vx.so
)