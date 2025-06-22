from setuptools import setup, find_packages

setup(
    name="DPro",
    description="Python library to preprocessing data",
    author="30team",
    author_email="balotsgar@gmail.com",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas~=2.2.3",
        "uvicorn~=0.34.3",
        "numpy~=2.1.3",
        "ydata-profiling~=4.16.1",
        "fastapi~=0.115.12",
        "pydantic~=2.11.4",
        "scikit-learn~=1.6.1",
        "natasha==1.5.0",
        "FLAML~=2.3.5",
        "scipy~=1.15.3"
    ]
)