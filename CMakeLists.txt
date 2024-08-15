# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(ctensor)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the header files
include_directories(include)

# Add the executable
add_executable(ctensor src/main.cpp src/tensor.cpp)

