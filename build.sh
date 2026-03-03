#!/bin/env bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

cmake -S. -Bbuild -DINPUT=$1 -GNinja
cmake --build build
./build/leet_out
