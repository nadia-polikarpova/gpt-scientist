.PHONY: all upload clean

all:
	python -m build

upload: all
	@set -a; \
	. .env; \
	twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info/