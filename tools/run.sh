#!/usr/bin/env bash

set -e
set -x
set -o pipefail

./tools/consistency_check.py
./tools/format.py
