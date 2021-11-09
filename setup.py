from setuptools import setup, find_packages

setup(
    name="ss2d",
    version='0.0.2',
    description='Simple Simulator 2D',
    author='tsuchiya-i',
    author_email='',
    install_requires=["opencv-python", "keras-rl2", "pillow"],
    packages=find_packages()
)
