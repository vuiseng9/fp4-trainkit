# setup.py
from setuptools import setup, find_packages

setup(
    name="fp4-training-kit",    
    version="0.1.0",                 
    packages=find_packages(),        
    install_requires=[               
        "torch",
        "tqdm"
    ],
    author="Vui Seng Chua",
    description="simulated fp4 training",
    python_requires=">=3.11",
)