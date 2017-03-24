from setuptools import setup

setup(name='stemgraphic',
      version='0.3.6',
      install_requires=[
          "docopt",
          "matplotlib",
          "pandas",
      ],
      scripts=['bin/stem','bin/stem.bat'],
      description='Graphic and text stem-and-leaf plots',
      url='http://github.com/fdion/stemgraphic',
      author='Francois Dion',
      author_email='francois.dion@gmail.com',
      license='MIT',
      packages=['stemgraphic'],
      zip_safe=False)
