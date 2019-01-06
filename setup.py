import io
import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

with io.open(os.path.join(here, 'VERSION')) as f:
    version = f.read()

setup(name='stemgraphic',
      version=version,
      install_requires=[
          "docopt",
          "matplotlib",
          "pandas",
          #"python-levenshtein",
          "seaborn"
      ],
      include_package_data=True,
      scripts=['bin/stem','bin/stem.bat'],
      description='Graphic and text stem-and-leaf plots',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/dionresearch/stemgraphic',
      author='Francois Dion',
      author_email='francois.dion@gmail.com',
      license='MIT',
      packages=['stemgraphic'],
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )
