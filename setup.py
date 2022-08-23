import setuptools

with open("README.md", "r") as fh:
    long_desc = fh.read()

setuptools.setup(
    name="pynomials",
    version="0.0.5",
    author="Divyajeet Singh",
    author_email="knightt1821@gmail.com",
    description="provides mathematical Polynomials as Python objects",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/divyajeettt/pynomials",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "matplotlib",
        "numpy",
    ]
)