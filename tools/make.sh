#!/usr/bin/env bash

set -e
set -x
set -o pipefail

# . /opt/intel/oneapi/setvars.sh
CXXFLAGS="-O3 -march=native -DNDEBUG -g -fno-omit-frame-pointer -Wall -Werror -Wformat -Wformat=2 -Wimplicit-fallthrough -Werror=format-security -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 -D_GLIBCXX_ASSERTIONS -fstrict-flex-arrays=3 -fstack-clash-protection -fstack-protector-strong"
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="$CXXFLAGS"
cmake --build build
./build/main -c ./example/config.toml -o ./example/output
