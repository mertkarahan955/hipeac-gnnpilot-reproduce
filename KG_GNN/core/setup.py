from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='KGGNN',
    ext_modules=[
        CUDAExtension(
        name='KGGNN', 
        sources=[
                    'src/KG_GNN.cpp',
                    'src/gnn_run.cpp',
                    'src/gnn_analysis.cpp',
                    'src/aggregate.cu',
                    'src/aggregate_sddmm.cu',
                    'src/aggregate_gat.cu',
                    'src/flash_partition.cu',
                    'src/aggregate_gin.cu',
                    'src/preprocessing.cu',
                    'src/bin_pack.cu'
                ],
        dlink=True,
        dlink_libraries=["dlink_lib"],
        extra_compile_args={'cxx':['-g'], 'nvcc':['-rdc=true']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })