from setuptools import setup, find_packages

setup(
    name="muvius",
    version="0.1.2",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.1.7",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "requests>=2.31.0",
        "pydantic>=2.5.2",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "rich>=13.7.0",
        "typer>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "muvius=muvius.cli:cli",
        ],
    },
    author="MUVIO AI",
    author_email="your.email@example.com",
    description="A framework for building and managing AI agents with memory systems and API endpoints",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tnbtinai/muvius-a2a-framework",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 