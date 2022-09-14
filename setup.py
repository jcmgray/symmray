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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='tensor block sparse symmetry autoray',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=[
        'autoray',
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx-autoapi',
            'myst-nb',
            'furo',
        ],
    },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/jcmgray/symmray/issues',
        'Source': 'https://github.com/jcmgray/symmray/',
    },
    include_package_data=True,
)
