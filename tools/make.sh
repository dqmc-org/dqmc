#!/usr/bin/env bash

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/dqmc -c ./example/config.toml -o ./example/output
./tools/compare.py ./tools/dos.out.1 ./example/output/dos.out
