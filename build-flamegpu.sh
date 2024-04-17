#!/usr/bin/bash
set -e

# FIXME: cmake should just find it..
#export PATH=/usr/local/cuda-12.2/bin:$PATH
# FIXME: do it inside "build-only-venv"
python3 -m pip install --user -r build-requirements.txt

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
  -DCMAKE_BUILD_TYPE=$build_type
  -DFLAMEGPU_VISUALISATION=OFF
  -DFLAMEGPU_BUILD_PYTHON=ON
  # needed for python agent_/device_function debuggin with cuda-gdb
  # see https://docs.flamegpu.com/guide/debugging-models/using-a-debugger.html#linux
  -DFLAMEGPU_RTC_EXPORT_SOURCES=ON
  #-DCMAKE_CUDA_COMPILER=nvcc                                            
  #-DCMAKE_CUDA_ARCHITECTURES="35;37;50;52;53;60;61;62;70;72;75;80;86"
)
if [[ "$no_seatbelts" != "" ]]; then
  cmake_opts+=(-DFLAMEGPU_SEATBELTS=OFF)
fi
cmake .. ${cmake_opts[@]}
ninja -t compdb > compilation_database.json
ninja pyflamegpu
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install ./lib/$build_type/python/dist/pyflamegpu-2*-linux_x86_64.whl
python3 -m pip install ipython pytest black ostruct
