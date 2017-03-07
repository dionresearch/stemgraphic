from setuptools import setup
with open('VERSION', 'r') as f:
    VERSION = f.read().rstrip('\n').replace("'", '')

setup(name='stemgraphic',
      version=VERSION,
      install_requires=[
          "docopt",
          "matplotlib",
          "pandas",
      ],
      scripts=['bin/stem'],
      description='Graphic and text stem-and-leaf plots',
      url='http://github.com/fdion/stemgraphic',
      author='Francois Dion',
      author_email='francois.dion@gmail.com',
      license='MIT',
      packages=['stemgraphic'],
      zip_safe=False)
