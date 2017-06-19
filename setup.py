from setuptools import find_packages, setup


setup(
    name='Teras',
    version='0.1.0',
    author='Hiroki Teranishi',
    author_email='teranishihiroki@gmail.com',
    description='framework for deep learning applications',
    url='https://github.com/chantera/teras',
    license='MIT',
    install_requires=['numpy', 'progressbar2', 'python-dateutil'],
    packages=find_packages(),
)