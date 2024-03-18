from setuptools import setup, find_packages

setup(
    name="softhand_sim",
    version="0.0.1",
    install_requires=["gymnasium>=0.26.0", "pygame==2.1.0"],
    packages=find_packages(),
)