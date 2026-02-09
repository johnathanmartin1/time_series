cmake_minimum_required(VERSION 4.2)

project(acf_cpp_backend)

set(Python_EXECUTABLE C:\\Users\\johna\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(pybind11_DIR C:\\Users\\johna\\anaconda3\\Library\\share\\cmake\\pybind11)

find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(acf_cpp_backend 
		 acf_cpp_backend.cpp
		 binding_avx_check.cpp
		 avx_check.cpp)