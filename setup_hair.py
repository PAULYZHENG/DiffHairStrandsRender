from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules = [
    CUDAExtension('hair_render.cuda.load_textures', [
        'hair_render/cuda/load_textures_cuda.cpp',
        'hair_render/cuda/load_textures_cuda_kernel.cu',
        ]),
    CUDAExtension('hair_render.cuda.create_texture_image', [
        'hair_render/cuda/create_texture_image_cuda.cpp',
        'hair_render/cuda/create_texture_image_cuda_kernel.cu',
        ]),
    CUDAExtension('hair_render.cuda.soft_rasterize', [
        'hair_render/cuda/soft_rasterize_cuda.cpp',
        'hair_render/cuda/soft_rasterize_cuda_kernel.cu',
        ]),
    CUDAExtension('hair_render.cuda.voxelization', [
        'hair_render/cuda/voxelization_cuda.cpp',
        'hair_render/cuda/voxelization_cuda_kernel.cu',
        ]),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "hair Rasterizer"',
    author='Shichen Liu',
    author_email='liushichen95@gmail.com',
    license='MIT License',
    version='1.0.0',
    name='hair_render',
    packages=['hair_render.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
