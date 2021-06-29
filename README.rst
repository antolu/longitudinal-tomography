.. image:: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/badges/master/pipeline.svg
.. image:: https://gitlab.cern.ch/longitudinaltomography/tomographyv3/badges/master/coverage.svg
    :target: https://gitlab.cern.ch/anlu/longitudinaltomography/-/jobs/artifacts/master/download?job=pages

Copyright 2020 CERN. This software is distributed under the terms of the
GNU General Public Licence version 3 (GPL Version 3), copied verbatim in
the file LICENCE.txt. In applying this licence, CERN does not waive the
privileges and immunities granted to it by virtue of its status as an
Intergovernmental Organization or submit itself to any jurisdiction.


.. contents:: Table of Contents


Installation
------------

The computationally intensive or time-critical parts of the library is
written in C++ and python bindings are provided using `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_.
The installation and usage of the library is the same for all operating systems, but
different dependencies are needed for different operating systems.

Prerequisites
=============

"""""
Linux
"""""

You need a C++ compiler like ``g++`` installed. This is not required if installing a prebuilt package from acc-py or pypi.

"""""""
Windows
"""""""

On Windows computers `MSVC >= 14.0 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools>`_
with the Windows 10 SDK is required.

In MinGW and WSL environments the standard ``g++`` compiler works out of the box.

"""""
MacOS
"""""

No offical tests have been done on MacOS, but presumably ``g++``, ``clang``/``llvm`` should work.

Install
=======

The Longitudinal Tomography package is available in prebuilt wheels for Python 3.6-3.9
on CERN Acc-Py and pypy.org as ``longitudinal-tomography``. The package can thus easily be installed on
a Linux machine using

::

    pip install longitudinal-tomography

The package can be installed on a MacOS or Windows machine in the same manner, but the
C++ extension will be built on install.

"""""""""""""""""""""
Other ways to install
"""""""""""""""""""""

**N.B.** If using conda, see the next section.

Clone the repository and run
::

   pip install .

The C++ extension will be built on install.


For development environments where it's preferable to compile the C++ extension inplace, it's possible to run the command
::

    pip install -e .

which will compile the C++ extension using the available compiler (decided by setuptools).

Manual build using conda
""""""""""""""""""""""""

Anaconda distributions come bundled with its own libgcc (like stdlibc++.so) which the compiled extension is automatically linked against.
Some Linux distrubitions like Arch Linux provide bleeding-edge GCC, which may compile the extension with newer than what is available in
the conda environment. This causes the extension to compile, but may produce various error messages when being imported.
To solve this, the user can install gcc in anaconda with

::

    conda install gcc_linux-64 gxx_linux-64

The C++ extension can then be compiled in place with

::

    python setup.py build_ext --inplace

Note that compiling with ``setup.py`` is considered legacy.

"""""""""""""""""""
Building with CMake
"""""""""""""""""""

The extension can be built with ``cmake`` wherever cmake is available. Building out of source is as simple as

::

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=../ ..
    make libtomo install

Experimental GPU support
------------------------

The tracking and reconstruction code currently has experimental support for GPU acceleration
using a custom written CUDA extension. The GPU CUDA extension provides the same interface as
that of the standard ``libtomo`` C++ pybind11 extension. The GPU extension replaces the
standard extension.

Currently the GPU extension can only be compiled and installed with CMake.

::

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=../ ..
    make libgputomo install
