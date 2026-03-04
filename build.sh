#!/bin/env bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <input_file> [-d]"
    exit 1
fi

DB="OFF"
if [ "$2" == "-d" ]; then
    DB="ON"
fi

cmake -S. -Bbuild -DINPUT=$1 -DDEBUG="$DB" -GNinja
cmake --build build
./build/leet_out
