install:
	pip install -e .


test:
	pytest ./tests/test_scripts.py -vv

fulltest:
	pytest ./ -vv

testd:
	pytest ./tests/test_scripts.py -s
