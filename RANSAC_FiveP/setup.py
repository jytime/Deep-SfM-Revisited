from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os.path import join

project_root = 'essential_matrix'
sources = [join(project_root, file) for file in ["essential_matrix.cu",
                                                 "essential_matrix_wrapper.cpp"]]

setup(
    name='essential_matrix',
    ext_modules=[
        CUDAExtension('essential_matrix',
            sources), # extra_compile_args, extra_link_args
        ],
    cmdclass={
        'build_ext': BuildExtension
    })

