#!/bin/bash

python3 ~/relcode_py/generate_fortran_input.py $1

taskset -c 8-15 ~/relcode/build/relcode.exe
