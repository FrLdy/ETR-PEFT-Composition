# Notes

## Ray Tune
- DDP set num_workers with number of GPUs availables, set num_gpus_per_worker=1

## Run eval on test each epoch end

- callback `on_epoch_end` before `maybe_log_save_evaluate` / solved, texts are logged with metrics


## Expes

- Models:
    - `meta-llama/Llama-3.1-8B`
    - `mistralai/Mistral-7B-v0.3`
    - `moussaKam/mbarthez`
