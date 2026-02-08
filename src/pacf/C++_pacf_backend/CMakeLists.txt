cmake_minimum_required(VERSION 4.2)

project(pacf_cpp_backend)

set(Python_EXECUTABLE C:\\Users\\johna\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(pybind11_DIR C:\\Users\\johna\\anaconda3\\Library\\share\\cmake\\pybind11)

find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(pacf_cpp_backend 
		 pacf_cpp_backend.cpp
		 avx_check.cpp)