#!/usr/bin/env python

from setuptools import setup, find_namespace_packages

setup(name = "addictif",
      version = "0.1",
      description = "ADDICTIF: ADvection-DIffusion-Chemistry in a Time-Independent Framework",
      author = "Paiman Shafabakhsh and Gaute Linga",
      author_email = "gaute.linga@mn.uio.no",
      url = 'https://github.com/gautelinga/addictif.git',
      classifiers = [
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python ',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = find_namespace_packages(where="src"),
      package_dir = {"": "src"},
      install_requires=[
          "argparse", 
          "h5py",
          "matplotlib",
          "mpi4py",
          "numpy",
          "scipy",
          "cppimport"
      ],
      entry_points = {"console_scripts": ["addictif=addictif.run_addictif:main"]},
      include_package_data=True,
      package_data={"addictif.chemistry": ["**/sols/*.dat"],
                    "addictif.common.fenicstools": ["cpp/*.cpp", "cpp/*.h"],
                    "addictif.common": ["helper_code.cpp"]},
    )