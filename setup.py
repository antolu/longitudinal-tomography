from setuptools import setup
from os import path
from glob import glob
import platform
from pybind11.setup_helpers import Pybind11Extension, build_ext

HERE = path.split(path.abspath(__file__))[0]
src_base = path.join('longitudinal_tomography', 'cpp_routines')

extra_compile_args = []
extra_link_args = []
if platform.system() == 'Windows':
    extra_compile_args.append('-openmp')
elif platform.system() == 'Linux':
    extra_compile_args.append('-fopenmp')
    extra_compile_args.append('-ffast-math')
    extra_link_args.append('-lgomp')


cpp_routines = Pybind11Extension(
    'longitudinal_tomography.cpp_routines.libtomo',
    cxx_std=17,
    sources=glob(path.join(src_base, 'src/libtomo/*.cpp')),
    include_dirs=[src_base],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args)

setup(cmdclass={'build_ext': build_ext}, ext_modules=[cpp_routines])
