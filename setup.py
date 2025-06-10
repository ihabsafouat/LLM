from setuptools import setup, find_packages

setup(
    name="gpt-from-scratch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pylzma",
        "tokenizers",
        "tqdm",
        "transformers",
        "numpy",
        "tqdm",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A PyTorch implementation of GPT from scratch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gpt-from-scratch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 