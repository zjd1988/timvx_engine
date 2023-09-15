#!/bin/bash
set -e
if [ "$1" == "A311D" ] || [ "$1" == "S905D3" ] || [ "$1" == "C308X" ];then
  COMPILE_TOOL_CHAIN=/data/zhaojd-a/gitee_codes/tool_chain/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu
elif [ "$1" == "C305X" ];then
  COMPILE_TOOL_CHAIN=/data/zhaojd-a/gitee_codes/tool_chain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf
else
  COMPILE_TOOL_CHAIN=
fi


ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build_for_$1
GCC_COMPILER=${COMPILE_TOOL_CHAIN}-gcc
GPP_COMPILER=${COMPILE_TOOL_CHAIN}-g++

if [ "$1" == "A311D" ] || [ "$1" == "S905D3" ] || [ "$1" == "C308X" ] || [ "$1" == "C305X" ];then
  TARGET_PLATFORM=$1
  echo "build for $1 target platform..."
else
  echo "unsupported $1 platform ..."
  exit 1
fi

if [ "$2" == "clean" ];then
  rm -r ${BUILD_DIR}
  rm -r ${ROOT_PWD}/3rd_party/TIM_VX
  exit 0
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake .. -DCMAKE_C_COMPILER=${GCC_COMPILER} \
         -DCMAKE_CXX_COMPILER=${GPP_COMPILER} \
         -DTARGET_PLATFORM=${TARGET_PLATFORM} \
         -DBUILD_PYTHON_LIB=OFF ${@:2}

#make -j`nproc`
#cd -
