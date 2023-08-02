cmake_minimum_required(VERSION 3.14)
project(timvx_engine)
include(ExternalProject)

set(TIM_VX_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rd_party/TIM_VX)
set(TIM_VX_INSTALL_DIR ${TIM_VX_SOURCE_DIR}/install)
ExternalProject_Add(TIM_VX
    PREFIX ${TIM_VX_SOURCE_DIR}
    GIT_REPOSITORY https://github.com/zjd1988/TIM-VX-python.git
    GIT_TAG "main"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${TIM_VX_INSTALL_DIR}
    # CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${TIM_VX_INSTALL_DIR} -DTIM_VX_ENABLE_VIPLITE=ON
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ")
set(TIM_VX_INCLUDE_PATH ${TIM_VX_INSTALL_DIR}/include)
set(TIM_VX_LIB_PATH ${TIM_VX_INSTALL_DIR}/lib)
set(JSON_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/3rd_party/nlohmann-json/include)

link_directories(${TIM_VX_LIB_PATH})

option(BUILD_SHARED_LIBS "build shared or static lib" ON)
file(GLOB_RECURSE TIMVX_ENGINE_SRC  ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if (BUILD_SHARED_LIBS)
    add_library(timvx_engine SHARED ${TIMVX_ENGINE_SRC})
else()
    add_library(timvx_engine STATIC ${TIMVX_ENGINE_SRC})
endif()
add_dependencies(timvx_engine TIM_VX)

target_include_directories(timvx_engine PUBLIC 
    ${TIM_VX_INCLUDE_PATH}
    ${JSON_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(timvx_engine PUBLIC tim-vx)