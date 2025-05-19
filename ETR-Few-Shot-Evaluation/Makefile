.PHONY: quality style test

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := tests src

# this target runs checks on all files

quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)


style:
	black --preview $(check_dirs)
	isort $(check_dirs)

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/
	python -c "import transformers; print(transformers.__version__)"

install-spacy-models:
	python -m spacy download fr_core_news_md
	python -m spacy download en_core_web_md

create-expe:
	cookiecutter cookicutter/experimentation -o experimentations/
