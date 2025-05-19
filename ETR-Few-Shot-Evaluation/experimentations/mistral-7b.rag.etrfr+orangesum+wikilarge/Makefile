rsync-results:
	rsync -avL -e ssh \
		--include="**/experiment_state-*.json" \
		--include="**/progress.csv" \
		--include="**/result.json" \
		--include="**/events.out.tfevents.*" \
		--include='*/' \
		--exclude="*" \
		${AUSTRAL_USER}@austral.criann.fr:~/ETR-Few-Shot-Evaluation$$(pwd | awk -F"ETR-Few-Shot-Evaluation" '{print $$NF}')/results .

hp-search:
	sbatch ./slurm.sh --gpus-per-node=1

