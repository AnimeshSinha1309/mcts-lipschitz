from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = list(map(lambda x: x.strip(), f.readlines()))

info = {
    "name": "montecomb",
    "version": "v0.1.0",
    "maintainer": "Animesh Sinha, Bhuvanesh Sridharan",
    "maintainer_email": "animesh.sinha@research.iiit.ac.in",
    "url": "https://github.com/AnimeshSinha1309/mcts-lipschitz/",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "entry_points": {"console_scripts": ["montecomb = montecomb:cli"]},
    "description": "montecomb is a library for comparing and contrasting monte carlo approaches for"
                   "decision over subsets, i.e. in combinatorial spaces.",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "provides": ["montecomb"],
    "install_requires": requirements,
    "package_data": {"qleet": ["tests/pytest.ini"]},
    "include_package_data": True,
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **info)
