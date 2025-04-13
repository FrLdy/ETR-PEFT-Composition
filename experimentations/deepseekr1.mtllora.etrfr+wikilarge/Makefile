rsync-results:
	rsync -avL -e ssh \
		--include="**/experiment_state-*.json" \
		--include="**/progress.csv" \
		--include="**/result.json" \
		--include="**/events.out.tfevents.*" \
		--include='*/' \
		--exclude="*" \
		${AUSTRAL_USER}@austral.criann.fr:~/etr-peft-composition$$(pwd | awk -F"etr-peft-composition" '{print $$NF}')/results .

hp-search:
	sbatch ./slurm.sh --gpus-per-node=4

inference:
	sbatch ./slurm.sh --gpus-per-node=1
