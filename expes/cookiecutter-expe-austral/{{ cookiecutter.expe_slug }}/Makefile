rsync-results:
	rsync -avL -e ssh \
		${AUSTRAL_USER}@austral.criann.fr:~/{{ cookiecutter.project_basename }}$$(pwd | awk -F"{{ cookiecutter.project_basename }}" '{print $$NF}')/results .

