from setuptools import setup, find_packages

setup(
    name="CRPN",
    version="1.0.0",
    author="hanluyt",
    author_email="hanluyt11@gmail.com",
    description="A Python package for CRPN model probing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hanluyt/CRPN",  
    packages=find_packages(),  
    install_requires=[
        "numpy>=1.21.0", 
        "torch>=2.3.0",
        "torchvision>=0.18.1",
        "pandas>=2.0.3",
        "opencv-python>=4.8.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)