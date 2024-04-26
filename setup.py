from __future__ import annotations

import os
import sys

import numpy as np
import setuptools
from Cython.Build import cythonize


include_dirs: list[str] = [np.get_include()]
compiler_directives: dict[str, int | bool] = {"language_level": 3}
define_macros: list[tuple[str, str | None]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
extra_link_args: list[str] = []
extra_compile_args: list[str] = []
purge = False

if "--production" in sys.argv:
    print("=" * 100, "compiling for production", "=" * 100)
    sys.argv.remove("--production")
    purge = True  # this flag will be used to remove the created .c files to minimize the docker image size


if tuple(sys.argv[1:3]) == ("clean", "--all"):
    # when switching between production and coverage we need to remove the _c.c file to
    # ensure that the cython code is recompiled with the correct compiler directives
    for file in ("nzthermo/_c.c", "nzthermo/_datetime.c"):
        print(f"removing {file}")
        if os.path.exists(file):
            os.remove(file)


if "--coverage" in sys.argv:
    if purge is True:
        raise ValueError("Cannot compile for coverage and production at the same time.")
    print("=" * 100, "compiling for coverage", "=" * 100)
    sys.argv.remove("--coverage")
    # in order to compile the cython code for test coverage  we need to include the following compiler directives...
    compiler_directives.update({"linetrace": True, "profile": True})
    # and include the following trace macros
    define_macros.extend([("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")])
else:
    # runtime:
    #   The schedule and chunk size are taken from the runtime scheduling variable, which can be set
    #   through the openmp.omp_set_schedule() function call, or the OMP_SCHEDULE environment variable.
    #   Note that this essentially disables any static compile time optimizations of the scheduling code
    #   itself and may therefore show a slightly worse performance than when the same scheduling policy
    #   is statically configured at compile time.
    # Dynamic scheduling:
    #  Is better when the iterations may take very different  amounts of time.
    if (omp := os.getenv("OMP_SCHEDULE", "dynamic")) not in ("static", "dynamic", "guided", "auto"):
        raise OSError(f"the provided ``OMP_SCHEDULE`` is invalid got := {omp}")
    os.environ["OMP_SCHEDULE"] = omp
    define_macros += [("OMP_SCHEDULE", omp)]
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]

extension_modules = [
    setuptools.Extension(
        "nzthermo._c",
        ["nzthermo/_c.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    setuptools.Extension(
        "nzthermo._datetime",
        ["nzthermo/_datetime.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setuptools.setup(
    ext_modules=cythonize(extension_modules, compiler_directives=compiler_directives),
)
