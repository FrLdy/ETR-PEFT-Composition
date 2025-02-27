rsync-results:
	rsync -avL -e ssh \
		--include="experiment_state-*.json" \
		--include="events.out.tfevents.*" \
		--exclude="*" \
		${AUSTRAL_USER}@austral.criann.fr:~/{{ cookiecutter.project_basename }}$$(pwd | awk -F"{{ cookiecutter.project_basename }}" '{print $$NF}')/results .

