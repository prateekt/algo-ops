conda_dev:
	conda env remove -n algo_ops_env
	conda env create -f conda.yaml

build:
	rm -rf dist
	rm -rf build
	rm -rf algo_ops.egg*
	python setup.py sdist bdist_wheel

deploy:
	twine upload dist/*

test:
	nose2

clean:
	rm -rf dist
	rm -rf build
	rm -rf algo_ops.egg*
	rm -rf .pytest_cache