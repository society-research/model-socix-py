# SociX Model

Implemented with [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2).

## Build & Run
```sh
./build-flamegpu.sh
source venv-Debug
pytest .
python sx.py
```

### Debugging
cuda-gdb requires the venv to copy the python executables, i.e. this setup
(default) is not sufficient:
```sh
$ ll
lrwxrwxrwx 1 ubuntu ubuntu    7 Apr  4 20:54 python -> python3
lrwxrwxrwx 1 ubuntu ubuntu   16 Apr  4 20:54 python3 -> /usr/bin/python3
lrwxrwxrwx 1 ubuntu ubuntu    7 Apr  4 20:54 python3.10 -> python3
```
To fix it, remove the symlink and copy the binary.
```sh
$ cd $VIRTUAL_ENV/bin
$ rm python3
$ cp /usr/bin/python3 ./
$ cuda-gdb --args python3 model.py -v
```
