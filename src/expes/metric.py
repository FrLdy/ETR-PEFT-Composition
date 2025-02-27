METRIC_KEY_TEXTS = "texts"


def compute_metrics(eval_pred, metrics_fn, tokenizer):
    eval_pred = tokenizer.eval_pred_manager(eval_pred)
    inputs, labels, predictions = (
        eval_pred.inputs,
        eval_pred.label_ids,
        eval_pred.predictions,
    )
    metrics = metrics_fn(
        predictions=predictions, references=labels, sources=inputs
    )

    metrics.update(
        {
            METRIC_KEY_TEXTS: {
                "inputs": inputs,
                "labels": labels,
                "predictions": predictions,
            },
        }
    )

    return metrics
