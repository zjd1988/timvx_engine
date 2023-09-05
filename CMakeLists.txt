cmake_minimum_required(VERSION 3.4.1)
project(pytimvx)

option(ENABLE_DEBUG_LOG "enable print debug log" ON)
option(BUILD_PYTHON_LIB "enable build python lib" ON)
option(BUILD_MODEL_TOOLS "enable build model tools lib" ON)
if(ENABLE_DEBUG_LOG)
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=1)
endif()

if(TIMVX_INSTALL_DIR)
    set(TIM_VX_INCLUDE_PATH ${TIMVX_INSTALL_DIR}/include)
    set(TIM_VX_LIB_PATH ${TIMVX_INSTALL_DIR}/lib)
else()
    set(TIM_VX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../host_build/install/include)
    set(TIM_VX_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../host_build/install/lib)
endif()

if(EXTERNAL_VIV_SDK)
    set(EXTERNAL_VIV_SDK_LIB_PATH ${EXTERNAL_VIV_SDK}/drivers)
    set(EXTERNAL_VIV_SDK_LIBS CLC GAL OpenVX OpenVXU VSC ArchModelSw NNArchPerf)
else()
    set(EXTERNAL_VIV_SDK_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../prebuilt-sdk/X86_64_linux/lib)
    set(EXTERNAL_VIV_SDK_LIBS CLC GAL OpenVX OpenVXU VSC ArchModelSw NNArchPerf Emulator)
endif()

set(PYBIND11_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/pybind11-2.9.2/include)
set(SPDLOG_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog-1.12.0/include)
set(NLOHMANN_JSON_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/nlohmann-json-3.11.2/include)
set(PYBIND11_JSON_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/pybind11_json/include)
set(CXXOPTS_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/cxxopts/include)
set(LIBNPY_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/libnpy/include)
set(STB_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/stb/include)
set(PROJECT_INCLUDE_PATH ${TIM_VX_INCLUDE_PATH} 
    ${SPDLOG_INCLUDE_PATH}
    ${NLOHMANN_JSON_INCLUDE_PATH}
    ${PYBIND11_JSON_INCLUDE_PATH}
    ${CXXOPTS_INCLUDE_PATH}
    ${LIBNPY_INCLUDE_PATH}
    ${STB_INCLUDE_PATH})
file(GLOB_RECURSE PYTIMVX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# build timvx engine lib
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-stack-protector -std=gnu99 -O2 -flax-vector-conversions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector -std=c++11 -O2")


link_directories(${TIM_VX_LIB_PATH})
link_directories(${EXTERNAL_VIV_SDK_LIB_PATH})
# build python lib
if(BUILD_PYTHON_LIB)
    find_package(Python REQUIRED COMPONENTS Development)
    set(CYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
    set(CYTHON_LIBS ${Python_LIBRARIES})
    set(PROJECT_LINK_LIBS ${PROJECT_LINK_LIBS} ${CYTHON_LIBS})
    message("use system python include: ${CYTHON_INCLUDE_DIRS}")
    message("use system python libs: ${CYTHON_LIBS}")

    add_library(pytimvx SHARED ${PYTIMVX_SRC})

    target_include_directories(pytimvx PRIVATE 
        ${PROJECT_INCLUDE_PATH}
        ${PYBIND11_INCLUDE_PATH}
        ${CYTHON_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

    target_link_libraries(pytimvx PRIVATE 
        tim-vx
        ${CYTHON_LIBS})
endif()

list(REMOVE_ITEM PYTIMVX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/pytimvx.cpp)
add_library(timvx_engine SHARED ${PYTIMVX_SRC})
target_include_directories(timvx_engine PRIVATE 
    ${PROJECT_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(timvx_engine PRIVATE tim-vx ${EXTERNAL_VIV_SDK_LIBS})

# build model tools
if(BUILD_MODEL_TOOLS)
    # common tools src
    set(COMMON_TOOLS_SRC 
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/timvx_model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/tool_utils.cpp)

    # model verify tool
    set(MODEL_VERIFY_TOOL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tools/model_verify.cpp)
    add_executable(model_verify ${MODEL_VERIFY_TOOL_SRC} ${COMMON_TOOLS_SRC})
    target_include_directories(model_verify PRIVATE 
        ${PROJECT_INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    target_link_libraries(model_verify PRIVATE timvx_engine tim-vx ${EXTERNAL_VIV_SDK_LIBS})

    # model infer tool
    set(MODEL_INFER_TOOL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tools/model_infer.cpp)
    add_executable(model_infer ${MODEL_INFER_TOOL_SRC} ${COMMON_TOOLS_SRC})
    target_include_directories(model_infer PRIVATE 
        ${PROJECT_INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    target_link_libraries(model_infer PRIVATE timvx_engine tim-vx ${EXTERNAL_VIV_SDK_LIBS})

    # model compile tool
    set(MODEL_COMPILE_TOOL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tools/model_compile.cpp)
    add_executable(model_compile ${MODEL_COMPILE_TOOL_SRC} ${COMMON_TOOLS_SRC})
    target_include_directories(model_compile PRIVATE 
        ${PROJECT_INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    target_link_libraries(model_compile PRIVATE timvx_engine tim-vx ${EXTERNAL_VIV_SDK_LIBS})

    # model benchmark tool
    set(MODEL_BENCHMARK_TOOL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tools/model_benchmark.cpp)
    add_executable(model_benchmark ${MODEL_BENCHMARK_TOOL_SRC} ${COMMON_TOOLS_SRC})
    target_include_directories(model_benchmark PRIVATE 
        ${PROJECT_INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    target_link_libraries(model_benchmark PRIVATE timvx_engine tim-vx ${EXTERNAL_VIV_SDK_LIBS})
endif()