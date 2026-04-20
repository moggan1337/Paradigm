"""
Setup file for Paradigm package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="paradigm-nas",
    version="0.1.0",
    author="Paradigm Contributors",
    author_email="contact@paradigm.ai",
    description="AutoML with Neural Architecture Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moggan1337/Paradigm",
    packages=find_packages(exclude=["tests", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "viz": [
            "networkx>=2.8.0",
            "tensorboard>=2.10.0",
        ],
        "distributed": [
            "torch>=1.9.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "networkx>=2.8.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "paradigm=paradigm.cli:main",
        ],
    },
)
