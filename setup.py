"""Build script for the `zane_cpp` pybind11 extension.

Usage:
    pip install -e .        # editable install, builds the C++ extension in place
    python setup.py build_ext --inplace

The rest of the project (the `zane` Python package) is pure Python and does
not require this extension to import — see zane/analytics_bridge.py for the
pure-Python fallback used when the compiled module isn't present.
"""
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "zane_cpp",
        sorted([
            "cpp/src/analytics.cpp",
            "cpp/bindings/bindings.cpp",
        ]),
        include_dirs=["cpp/include"],
        cxx_std=17,
    ),
]

setuptools.setup(
    name="zane-digital-mind",
    version="1.0.0",
    description="Zane's digital mind: a hybrid C++/Python Nindroid cognitive architecture",
    packages=setuptools.find_packages(include=["zane", "zane.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
