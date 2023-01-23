from setuptools import setup, find_packages

setup(
    name = 'cobamp',
    version = '0.2.1',
    package_dir = {'':'src'},
    packages = find_packages('src'),
    install_requires = ["indexed==1.2.1",
                        "pandas==1.3.5",
                        "scipy==1.7.3",
                        "numpy==1.21.6",
                        "matplotlib==3.5.2",
                        "optlang==1.5.2",
                        "pathos==0.2.9",
                        "boolean.py==4.0"],

    author = 'Vítor Vieira',
    author_email = 'vvieira@ceb.uminho.pt',
    description = 'cobamp - pathway analysis methods for genome-scale metabolic models',
    license = 'GNU General Public License v3.0',
    keywords = 'pathway analysis metabolic model',
    url = 'https://github.com/BioSystemsUM/cobamp',
    long_description = open('README.rst').read(),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
