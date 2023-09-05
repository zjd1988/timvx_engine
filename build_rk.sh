#!/bin/bash
set -e
if [ "$1" == "arm64" ];then
  COMPILE_TOOL_CHAIN=/data/zhaojd-a/gitee_codes/tool_chain/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu
elif [ "$1" == "arm32" ];then
  COMPILE_TOOL_CHAIN=/data/zhaojd-a/gitee_codes/tool_chain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf
else
  COMPILE_TOOL_CHAIN=
fi

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
if [ "$1" == "arm32" ];then
  BUILD_DIR=${ROOT_PWD}/build_for_arm32
  TARGET_PLATFORM=arm32
  GCC_COMPILER=${COMPILE_TOOL_CHAIN}-gcc
  GPP_COMPILER=${COMPILE_TOOL_CHAIN}-g++
  echo "build for arm target platform..."
elif [ "$1" == "arm64" ];then
  BUILD_DIR=${ROOT_PWD}/build_for_arm64
  TARGET_PLATFORM=arm64
  GCC_COMPILER=${COMPILE_TOOL_CHAIN}-gcc
  GPP_COMPILER=${COMPILE_TOOL_CHAIN}-g++
  echo "build for arm64 target platform..."
else
  echo "wrong input parameter..."
  exit 1
fi

if [ "$2" == "clean" ];then
  rm -r ${BUILD_DIR}
  exit 0
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

TIMVX_INSTALL_DIR=$2
EXTERNAL_VIV_SDK=$3

cd ${BUILD_DIR}
cmake .. -DCMAKE_C_COMPILER=${GCC_COMPILER} \
         -DCMAKE_CXX_COMPILER=${GPP_COMPILER} \
         -DTIMVX_INSTALL_DIR=${TIMVX_INSTALL_DIR} \
         -DEXTERNAL_VIV_SDK=${EXTERNAL_VIV_SDK} \
         -DBUILD_PYTHON_LIB=OFF ${@:2}

make -j`nproc/2` 
cd -