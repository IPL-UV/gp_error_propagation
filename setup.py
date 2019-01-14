from distutils.core import setup

setup(
    name="gp_error_propagation",
    version="0.1",
    description="Functions to operate code in error propagation paper.",
    author="Juan Emmanuel Johnson",
    author_email="jemanjohnson34@gmail.com",
    install_requires=[
        "numpy >= 1.16.0", 
        "scikit-learn >= 0.20.2",
        "scipy >= 1.2.0", 
        "matplotlib >= 3.0.0",
        "numba >= 0.42.0"
    ],
    long_description=open('README.md').read(),
)