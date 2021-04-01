# Basic machine learning framework

A simple implementation of gradient descend for neural networks.

## Prerequisites

In case you would like to try my implementation yourself you need:

- CMake 3.10 and higher
- A C++ compiler that supports C++11 and higher

## Build

### Debian and Ubuntu

```shell
$ sudo apt install cmake build-essential
$ git clone https://github.com/DerBeischl/basic_ml.git
$ cd basic_ml
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ ./xor # Run xor example
```

### Arch

```shell
$ sudo pacman -S cmake base-devel
$ git clone https://github.com/DerBeischl/basic_ml.git
$ cd basic_ml
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ ./xor # Run xor example
```

# License
