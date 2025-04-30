from setuptools import setup, find_packages

setup(
    name="echoml",
    version="0.1.0",
    description="A machine learning utility for analyzing echograms using K-Means cluster maps built on echopype",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Michael C Ryan",
    author_email="spacetime.engineer@gmail.com",
    url="https://github.com/mryan/echoml",
    packages=find_packages(where=".", exclude=["tests", "examples"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "xarray",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "echopype>=0.6.0",
        "echoregions",
        "loguru",
    ],
    #entry_points={
    #    "console_scripts": [
    #        "kmc=cond.src.cond:main",  # Maps `cond` command to the `main` function in `cond.py`
    #    ],
    #},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "aa-kmap=echoml.parser:main"
        ],
    },
)