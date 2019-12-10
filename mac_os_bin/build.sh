#!/bin/bash

rm -rf cmake_build
rm -rf ./bin/*
mkdir cmake_build
cd cmake_build
cmake ..
make

sleep 1
./SolveHomography
cd ..
