from setuptools import setup, find_packages

setup(
    name="aurora-rca",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # Read from requirements.txt
    ],
    author="Jaykumar Maheshkar",
    author_email="jay.maheshkar@usbank.com",
    description="Autonomous Root Cause Analysis for Microservices",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jay-gatech/agenticai-rca-langchain",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)
