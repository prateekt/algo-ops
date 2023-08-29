conda_dev:
	conda env remove -n algo_ops_env
	conda env create -f conda.yaml

build:
	rm -rf dist
	rm -rf build
	rm -rf algo_ops.egg*
	python3 -m pip install --upgrade build
	python3 -m build

deploy:
	python3 -m pip install --upgrade twine
	twine upload dist/*

test:
	nose2

clean:
	rm -rf dist
	rm -rf build
	rm -rf algo_ops.egg*
	rm -rf .pytest_cache