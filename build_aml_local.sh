#!/bin/bash
set -e

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build_for_$1
GCC_COMPILER=gcc
GPP_COMPILER=g++

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
