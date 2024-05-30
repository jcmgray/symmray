from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='symmray',
    version='0.0.1',
    description='A minimal block sparse symmetric tensor python library',
    long_description=long_description,
    url='https://github.com/jcmgray/symmray',
    author='Johnnie Gray',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='tensor block sparse symmetry autoray',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=[
        'autoray',
    ],
    extras_require={
        "tests": [
            "numpy",
            "coverage",
            "pytest",
            "pytest-cov",
        ],
        'docs': [
            'sphinx>=2.0',
            'sphinx-autoapi',
            'astroid<3',
            'sphinx-copybutton',
            'myst-nb',
            'furo',
            'setuptools_scm',
            'ipython!=8.7.0',
        ],
    },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/jcmgray/symmray/issues',
        'Source': 'https://github.com/jcmgray/symmray/',
    },
    include_package_data=True,
)
