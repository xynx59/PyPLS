import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyPLS",
    version="0.0.1",
    author="Renjian Pan and Xingyue Ding",
    author_email="xynx29@gmail.com",
    description="A function for PLSR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xynx59/PyPLS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)