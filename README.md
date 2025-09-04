# dqmc-framework
![workflow](https://github.com/dqmc-org/dqmc/actions/workflows/ci.yml/badge.svg?branch=master)

**This is a fork of the original [dqmc-framework](https://github.com/JefferyWangSH/dqmc-framework) by JefferyWangSH.**

In this repository, we present a general C++ implementation of determinant Quantum Monte Carlo `(DQMC)` algorithm for the simulation of fermionic quantum models on various lattice geometries.

Currently, simulations of the fermionic Hubbard model with both attractive and repulsive interaction are supported. Different lattices, e.g. 2d square lattice, 2d honeycomb lattice (todo) and 3d cubic lattice (todo) are supported or scheduled to be implemented.

<img src="./assets/plot.png" width="450">


---

## Installation ##

### Prerequisites ###

* `gcc/g++` `( version >= 7.1, support C++17 standard )` and `cmake` `( version >= 3.21 )` installed.
* `Boost C++ libraries` `( version >= 1.71 )` installed.
* `Eigen library` `( version >= 3.4.0 )` providing a user-friendly matrix interfaces.

### Building ###

1. Configure the build with CMake:
   ```shell
   cmake -S . -B build -G Ninja
   ```

2. Build the project:
   ```shell
   cmake --build build
   ```

### Usage ###

Run the simulation with default parameters:
```shell
./build/main
```

Or use command-line options:
```shell
./build/main --help  # See all available options
```

For detailed configuration options and examples, see [example/README.md](example/README.md).

<!-- 1. Download the source code from github.
    ``` shell
    $ # download the source code
    $ git clone https://github.com/JefferyWangSH/general-dqmc.git {PROGRAM_ROOT}
    ```
2. Enter the [`build/`](build/) directory and run [`runcmake.sh`](build/runcmake.sh) which will analyse the program using cmake.
    ``` shell
    $ # initialize cmake
    $ cd {PROGRAM_ROOT}/build && ./runcmake.sh
    ```
3. Enter the [`run/`](run/) directory and compile the codes. 
    ``` shell
    $ # build the program
    $ cd {PROGRAM_ROOT}/run && ./make.sh
    ```
4. Run the script [`run/batch.sh`](run/batch.sh) to start the simulation if the program is successfully build. ( We use `Slurm` for managements of program tasks, hence the simulation parameters should be edited in the script [`run/batch.sh`](run/batch.sh) in advance. )
    ```shell
    $ # edit the simulation params
    $ vim batch.sh

    $ # start simulation using Slurm
    $ ./batch.sh
    ```
5. Running the program directly with command `mpirun` also works, and one can always use option `--help` to see helping messages:
    ``` shell
    $ # start simulation
    $ mpirun -np 4 --oversubscribe {PROGRAM_ROOT}/build/dqmc
    $
    $ # show helping messages
    $ mpirun {PROGRAM_ROOT}/build/dqmc --help
    ``` -->


## Features ##

1. **Core DQMC modules independent of specific models and lattices.** The well-designed DQMC sweeping process is tested stable and of high computational efficiency.
2. **Modularly designed and highly extensible.** To simulate other models on different lattices, one should simply write their own Model and Lattice class, which should be derived from corresponding base classes, and implement the virtual interfaces and methods in a correct way. 
3. Support equal-time and dynamical measurements of various physical observables, and **adding measurements of user-defined observables is straightforward.**
4. Support checkerboard break-ups with efficient linear algebra ( for bipartite lattices only ).


## References ##

1. H. Fehske, R. Schneider, A. Weiße, Computational Many-Particle Physics, *Springer*, 2008. [doi](https://doi.org/10.1007/978-3-540-74686-7)
2. James Gubernatis, Naoki Kawashima, Philipp Werner, Quantum Monte Carlo Methods: Algorithms for Lattice Models, *Cambridge University Press*, 2016. [doi](https://doi.org/10.1017/CBO9780511902581)
3. Xiao Yan Xu, *Tutorial*: Solving square lattice Hubbard Model with DQMC, 2016. [doi](http://ziyangmeng.iphy.ac.cn/files/teaching/SummerSchoolSimpleDQMCnoteXYX201608.pdf)
4. C. N. Varney, C. R. Lee, Z. J. Bai et al., Quantum Monte Carlo study of the two-dimensional fermion Hubbard model, *Phys. Rev. B 80, 075116*, 2009. [doi](https://doi.org/10.1103/PhysRevB.80.075116)
5. Thereza Paiva, Raimundo R. dos Santos, R. T. Scalettar et al., Critical temperature for the two-dimensional attractive Hubbard model, *Phys. Rev. B 69, 184501*, 2004. [doi](https://doi.org/10.1103/PhysRevB.69.184501)
6. Xiao Yan Xu, Kai Sun, Yoni Schattner et al., Non-Fermi Liquid at (2 + 1) D Ferromagnetic Quantum Critical Point, *Phys. Rev. X 7, 031058*. [doi](https://doi.org/10.1103/PhysRevX.7.031058)
7. Douglas J. Scalapino, Steven R. White, and Shoucheng Zhang, Insulator, metal, or superconductor: The criteria, *Phys. Rev. B 47, 7995.* [doi](https://doi.org/10.1103/PhysRevB.47.7995)


## License & Support ##

This code is an open-source software under the MIT license.

If any problems or bugs, feel free to contact me via email 17307110117@fudan.edu.cn.