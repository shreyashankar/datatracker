from setuptools import setup, find_packages

setup(
    name='example',
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'pandas',
        'numpy',
        'pylint',
        'mypy',
        'scikit-learn'
    ]
)