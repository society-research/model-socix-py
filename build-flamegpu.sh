#!/usr/bin/bash
set -e

python3 -m venv build-venv
source build-venv/bin/activate
python3 -m pip install -r build-requirements.txt

git submodule update --init --recursive

#build_type=Release
build_type=Debug
#no_seatbelts=-no_seatbelts
no_seatbelts=""
build_dir=third_party/FLAMEGPU2/build-$build_type$no_seatbelts
[[ $CLEAN == true ]] && rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir
cmake_opts=(
  -G Ninja

  # basic build config
  -DCMAKE_BUILD_TYPE=$build_type
  -DFLAMEGPU_VISUALISATION=OFF
  -DFLAMEGPU_BUILD_PYTHON=ON

  # increases agent functino compile time by ~60s
  #-DFLAMEGPU_USE_GLM=ON # enable math: https://github.com/g-truc/glm

  # needed for python agent_/device_function debuggin with cuda-gdb
  # see https://docs.flamegpu.com/guide/debugging-models/using-a-debugger.html#linux
  -DFLAMEGPU_RTC_EXPORT_SOURCES=ON
  # enable `#include "header.cuh"` for RTC agent functions
  -DFLAMEGPU_RTC_INCLUDE_DIRS=.

  # gpu config
  -DCMAKE_CUDA_COMPILER=nvcc
  #-DCMAKE_CUDA_ARCHITECTURES="75" # <- l01 gpu
  #-DCMAKE_CUDA_ARCHITECTURES="50;52;53;60;61;62;70;72;75;80;86;87;89;90"
  #-DCMAKE_CUDA_ARCHITECTURES_ALL="35;37;50;52;53;60;61;62;70;72;75;80;86;87"
)
if [[ "$no_seatbelts" != "" ]]; then
  cmake_opts+=(-DFLAMEGPU_SEATBELTS=OFF)
fi
cmake .. ${cmake_opts[@]}
ninja -t compdb > compilation_database.json
ninja pyflamegpu
deactivate # build-venv
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install ./lib/$build_type/python/dist/pyflamegpu-2*-linux_x86_64.whl
cd -
python3 -m pip install -r requirements.txt
