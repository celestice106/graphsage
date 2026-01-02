"""
Setup script for GraphSAGE training package.
"""

from setuptools import setup, find_packages

setup(
    name="graphsage_training",
    version="1.0.0",
    description="GraphSAGE training for Memory R1 structural embeddings",
    author="Memory R1 Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "networkx>=3.0",
        "numpy>=1.23.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "matplotlib>=3.7.0",
            "tqdm>=4.65.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "graphsage-train=scripts.train:main",
            "graphsage-generate=scripts.generate_data:main",
            "graphsage-evaluate=scripts.evaluate:main",
        ],
    },
)
