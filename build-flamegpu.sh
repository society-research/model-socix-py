#!/usr/bin/bash
set -e

# TODO: cmake should just find it..
export PATH=/usr/local/cuda-12.2/bin:$PATH
# TODO: do it inside venv
python3 -m pip install -r requirements.txt

[[ $CLEAN == true ]] && rm -rf third_party/FLAMEGPU2/build
mkdir -p third_party/FLAMEGPU2/build
cd third_party/FLAMEGPU2/build
cmake_opts=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DFLAMEGPU_VISUALISATION=ON
  -DFLAMEGPU_BUILD_PYTHON=ON
)
cmake .. ${cmake_opts[@]}
ninja
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install ./lib/Release/python/dist/pyflamegpu-2*-linux_x86_64.whl
python3 -m pip install ipython
