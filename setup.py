import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuromod",
    version="0.0.1",
    author="Rodrigo Carrasco",
    author_email="rodrigo.carrasco.davis@gmail.com",
    description="maximizing value function",
    url="https://github.com/rodrigcd/neuromod",
    packages=setuptools.find_packages(),
)