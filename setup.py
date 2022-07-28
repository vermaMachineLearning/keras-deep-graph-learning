from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='keras_dgl',
      version='1.0',
      description='Python package for Keras Deep Learning on Graphs',
      long_description = long_description,
      url='https://github.com/chernyavskaya/keras-deep-graph-learning',
      author='Saurabh Verma',
      license='MIT',
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3"],
      python_requires=">=3.6"
)
