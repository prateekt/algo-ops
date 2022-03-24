build:
	rm -rf dist
	rm -rf build
	rm -rf algo_ops.egg*
	python setup.py sdist bdist_wheel

deploy:
	twine upload dist/*

test:
	nosetests