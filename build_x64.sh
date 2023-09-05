#!/bin/bash
set -e

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
# build x64
BUILD_DIR=${ROOT_PWD}/build_for_x64
TARGET_PLATFORM=x64
GCC_COMPILER=gcc
GPP_COMPILER=g++
echo "build for x64 target platform..."

if [ "$1" == "clean" ];then
  rm -r ${BUILD_DIR}
  exit 0
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

TIMVX_INSTALL_DIR=$1

cd ${BUILD_DIR}
cmake .. -DCMAKE_C_COMPILER=${GCC_COMPILER} \
         -DCMAKE_CXX_COMPILER=${GPP_COMPILER} \
         -DTIMVX_INSTALL_DIR=${TIMVX_INSTALL_DIR} \
         -DTARGET_PLATFORM=${TARGET_PLATFORM} ${@:2}

make -j`nproc/2` 
cd -