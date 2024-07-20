
from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    install_requires = [line for line in f]

packages = [a for a in find_namespace_packages(where='.') if a[:6]=='deepSI_lite']

setup(name = 'deepSI_lite',
      version = '0.1.0',
      description = 'Deep Dynamical Model Estiamtion',
      author = 'Gerben I. Beintema',
      author_email = 'g.i.beintema@tue.nl',
      license = 'BSD 3-Clause License',
      python_requires = '>=3.6',
      packages=packages,
      install_requires = install_requires,
      extras_require = dict(
        docs = ['sphinx>=1.6','sphinx-rtd-theme>=0.5']
        )
    )
