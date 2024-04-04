#!/usr/bin/bash
set -e

# TODO: cmake should just find it..
export PATH=/usr/local/cuda-12.2/bin:$PATH
# TODO: do it inside venv
python3 -m pip install -r requirements.txt

git submodule update --init --recursive

#build_type=Release
build_type=Debug
no_seatbelts=-no_seatbelts
#no_seatbelts=""
build_dir=third_party/FLAMEGPU2/build-$build_type$no_seatbelts
[[ $CLEAN == true ]] && rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir
cmake_opts=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=$build_type
  -DFLAMEGPU_VISUALISATION=OFF
  -DFLAMEGPU_BUILD_PYTHON=ON
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
python3 -m pip install ipython
