rsync-results:
	rsync -avL -e ssh \
		--include="experiment_state-*.json" \
		--include="events.out.tfevents.*" \
		--exclude="*" \
		${AUSTRAL_USER}@austral.criann.fr:~/etr-peft-composition$$(pwd | awk -F"etr-peft-composition" '{print $$NF}')/results .

