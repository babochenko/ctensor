# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(ctensor)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES
  src/criterion.cpp
  src/tensor.cpp
  src/main.cpp
)

set(HEADERS
  src/backward.h
  src/criterion.h
  src/functional.h
  src/tensor.h
)

# Add the executable
add_executable(ctensor ${SOURCES})

