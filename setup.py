import setuptools

# Load the long_description from README.md
with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TsaLib",
    version="0.0.2",
    author="Olivier Morel",
    author_email="molivier933@gmail.com",
    description="Personal package for the TSA course in HEIG-VD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/morel-olivier/tsaLib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
