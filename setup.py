from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='comm_cuda',
    ext_modules=[
        CUDAExtension('comm_cuda', [
            'comm_cuda.cpp',
            'comm_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
