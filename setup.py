from setuptools import setup

setup(name='crypto_clustering',
	version ='0.1',
	description = 'Package for clustering analysis on cryptocurrency prices',
	url = 'https://github.com/jacquelinegarrahan/crypto_clustering',
	author = 'Jacqueline Garrahan',
	author_email = 'jacquelinegarrahan@protonmail.com',
	license= 'GNU GENERAL PUBLIC LICENSE',
	packages = ['crypto_clustering'],
	install_requires= ['matplotlib', 'scikit-learn', 'sklearn', 'gdax', 'poloniex', 'scipy', 'numpy', 'pyparsing', 'six==1.10.0', 'requests==2.13.0'],
	zip_safe=False)
