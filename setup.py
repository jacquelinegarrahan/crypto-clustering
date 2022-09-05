from setuptools import setup, find_packages

from os import path

cur_dir = path.abspath(path.dirname(__file__))

# parse requirements
with open(path.join(cur_dir, "requirements.txt"), "r") as f:
    requirements = f.read().split()


setup(name='crypto_clustering',
	description = 'Package for clustering analysis on cryptocurrency prices',
	url = 'https://github.com/jacquelinegarrahan/crypto-clustering',
	author = 'Jacqueline Garrahan',
	author_email = 'jacquelinegarrahan@protonmail.com',
	license= 'GNU 3',
	packages = find_packages(),
	install_requires=requirements
)
