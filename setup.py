import io
import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

with io.open(os.path.join(here, 'VERSION')) as f:
    version = f.read()

setup(name='stemgraphic',
      version=version,
      install_requires=[
          "docopt",
          "matplotlib",
          "pandas",
          "python-levenshtein",
          "seaborn"
      ],
      scripts=['bin/stem','bin/stem.bat'],
      description='Graphic and text stem-and-leaf plots',
      url='http://github.com/fdion/stemgraphic',
      author='Francois Dion',
      author_email='francois.dion@gmail.com',
      license='MIT',
      packages=['stemgraphic'],
      zip_safe=False)
