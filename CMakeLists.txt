cmake_minimum_required(VERSION 3.10.2)
project(MachineLearning VERSION 1.0 DESCRIPTION "Collection of tools for basic ML")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
    CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo"
    FORCE)

set(CMAKE_CXX_FLAGS_DEBUG
    "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1 -Wall -Wextra"
    CACHE STRING "Flags used by the C++ compiler during AddressSanitizer builds."
    FORCE)

set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -Wall -Wextra"
    CACHE STRING "Flags used by the C++ compiler during Release builds."
    FORCE)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "-O3 -g -Wall -Wextra"
    CACHE STRING "Flags used by the C++ compiler during RelWithDebInfo builds."
    FORCE)

set(sources
    ./src/layer.cc
    ./src/matrix.cc
    ./src/random.cc
    ./src/network.cc
    ./src/activations.cc
)

add_library(MBL SHARED "${sources}")
set_target_properties(MBL PROPERTIES VERSION ${PROJECT_VERSION})
target_include_directories(MBL PRIVATE ./include)

add_executable(simple ./app/simple.cc)
target_include_directories(simple PRIVATE ./include)
target_link_libraries(simple MBL)

add_executable(xor ./app/xor.cc)
target_include_directories(xor PRIVATE ./include)
target_link_libraries(xor MBL)
