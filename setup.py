from setuptools import setup, find_packages

setup(
    name="eum-classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "lightfm",
        "numpy",
        "joblib",
        "tqdm",
    ],
) 