cmake_minimum_required(VERSION 3.26)
project(CPP)

set(CMAKE_CXX_STANDARD 17)

set(pybind11_DIR pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)

pybind11_add_module(simulator SHARED main.cpp Simulator.cpp)
