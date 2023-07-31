cmake_minimum_required(VERSION 3.14)
project(timvx_engine)
include(ExternalProject)

set(TIM_VX_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rd_party/TIM_VX)
set(TIM_VX_BUILD_DIR ${TIM_VX_SOURCE_DIR}/host_build)
set(TIM_VX_INSTALL_DIR ${TIM_VX_BUILD_DIR}/install)
ExternalProject_Add(TIM_VX
    PREFIX ${TIM_VX_SOURCE_DIR}
    GIT_REPOSITORY https://github.com/zjd1988/TIM-VX-python.git
    GIT_TAG "main"
    GIT_PROCESS ON
    SOURCE_DIR ${TIM_VX_SOURCE_DIR}
    BINARY_DIR ${TIM_VX_BUILD_DIR}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${TIM_VX_INSTALL_DIR}
)

# include(FetchContent)
# FetchContent_Declare(
#     TIM_VX
#     GIT_REPOSITORY https://github.com/zjd1988/TIM-VX-python.git
#     GIT_PROCESS ON
#     GIT_TAG "main"
#     GIT_SHALLOW ON
# )
# FetchContent_GetProperties(TIM_VX)
# if(NOT TIM_VX_POPULATED)
#     FetchContent_Populate(TIM_VX)
#     add_subdirectory(${TIM_VX_SOURCE_DIR} ${TIM_VX_BINARY_DIR})
# endif()
# FetchContent_MakeAvailable(TIM_VX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ")
set(TIM_VX_INCLUDE_PATH ${TIM_VX_INSTALL_PATH}/include)
set(TIM_VX_LIB_PATH ${TIM_VX_INSTALL_PATH}/lib)
set(JSON_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/3rd_party/nlohmann-json/include)

link_directories(${TIM_VX_LIB_PATH})

option(BUILD_SHARED_LIBS "build shared or static lib" ON)
file(GLOB_RECURSE SRC  ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if (BUILD_SHARED_LIBS)
    add_library(timvx_engine SHARED ${SRC})
else()
    add_library(timvx_engine STATIC ${SRC})
endif()
# add_dependencies(timvx_engine TIM_VX)

target_include_directories(timvx_engine PUBLIC 
    ${TIM_VX_INCLUDE_PATH}
    ${JSON_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(timvx_engine PUBLIC tim-vx)