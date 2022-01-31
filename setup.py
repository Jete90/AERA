from setuptools import setup, find_packages


setup(
    name="aera",
    version="1.0",
    packages=find_packages(),
    # metadata to display on PyPI
    author="Climate and Environmental Physics, University of Bern",
    author_email="jens.terhaar@unibe.ch",
    description=(
        "Implementation of the AERA algorithm (see Terhaar et al., in review)"),
    license="CC BY-NC-SA 4.0",
    keywords="research, science, climate change, mitigation",
    project_urls={
        "Source Code": "https://github.com/Jete90/AERA",
    },
    package_data={
        'aera': ['data/*.dat'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'netCDF4',
        'scipy',
        'dask[complete]',
        'xarray',
    ],
)
