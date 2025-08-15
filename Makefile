.PHONY: all upload clean

all:
	python -m build

upload: all
	twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info/