from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler
import os
import sys
import shlex

HERE = os.path.dirname(os.path.abspath(__file__))
output_dir = HERE

extra_compile_args = []
extra_link_args = []

if sys.platform != "win32":
    extra_compile_args = [
        "-fPIC",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-unreachable-code",
    ]

archflags = shlex.split(os.environ.get("ARCHFLAGS", ""))
if archflags:
    extra_compile_args.extend(archflags)
    extra_link_args.extend(archflags)

c_extension = Extension(
    name="libmatching_placeholder",
    sources=[
        os.path.join(HERE, "bottleneckBipartiteMatching.c"),
        os.path.join(HERE, "matrixUtils.c"),
        os.path.join(HERE, "mmio.c"),
        os.path.join(HERE, "extern", "cheap.c"),
        os.path.join(HERE, "extern", "matching.c"),
    ],
    include_dirs=[os.path.join(HERE, "extern")],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=["m"] if sys.platform != "win32" else [],
)


class BuildSharedLibrary(build_ext):
    
    def run(self) -> None:
        """
        Build only the PPSU shared library.

        Why this exists
        ---------------
        This project needs a plain shared library for ctypes, not a Python
        extension module with setuptools' normal filename/copy semantics.
        Overriding ``run`` avoids setuptools attempting to copy a nonexistent
        placeholder extension artifact after the custom shared library has
        already been built.
        """
        self.compiler = new_compiler(
            compiler=self.compiler,
            verbose=self.verbose,
        )
        customize_compiler(self.compiler)
        self.build_shared_library()

    def build_shared_library(self) -> None:
        for ext in self.extensions:
            objects = self.compiler.compile(
                ext.sources,
                include_dirs=ext.include_dirs,
                extra_postargs=ext.extra_compile_args,
            )

            lib_name = "libmatching_for_PPSU.dll" if sys.platform == "win32" else "libmatching_for_PPSU.so"
            lib_path = os.path.join(output_dir, lib_name)
            self.compiler.link_shared_object(
                objects,
                lib_path,
                libraries=ext.libraries,
                extra_postargs=getattr(ext, "extra_link_args", []),
            )
            print(f"Built shared library: {lib_path}")


setup(
    name="libmatching_for_PPSU",
    version="0.1",
    ext_modules=[c_extension],
    cmdclass={"build_ext": BuildSharedLibrary},
)