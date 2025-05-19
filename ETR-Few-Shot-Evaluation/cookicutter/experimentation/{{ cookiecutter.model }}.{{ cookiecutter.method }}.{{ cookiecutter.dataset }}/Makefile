rsync-results:
	rsync -avL -e ssh \
		--include="**/experiment_state-*.json" \
		--include="**/progress.csv" \
		--include="**/result.json" \
		--include="**/events.out.tfevents.*" \
		--include='*/' \
		--exclude="*" \
		${AUSTRAL_USER}@austral.criann.fr:~/{{ cookiecutter.project_basename }}$$(pwd | awk -F"{{ cookiecutter.project_basename }}" '{print $$NF}')/results .

hp-search:
	sbatch ./slurm.sh --gpus-per-node={{ cookiecutter.gpus }}

