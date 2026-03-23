from setuptools import setup, find_packages

setup(
    name="arc-active-inference",
    version="0.1.0",
    description="Neurosymbolic ARC-AGI solver using Active Inference and Global Workspace Theory",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.7.0",
        ],
    },
)
