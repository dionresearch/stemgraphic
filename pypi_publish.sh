#!/usr/bin/env bash
#python3 setup.py register -r pypitest
#python3 setup.py sdist upload -r pypitest

pandoc -f markdown -t rst README.md > README.rst
python3 setup.py register
python3 setup.py sdist upload
