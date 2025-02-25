from functools import partial, reduce

import evaluate
import nltk
import numpy as np
import spacy
import textacy.extract
import textacy.preprocessing
import textdescriptives as td
from datasets.utils.py_utils import Literal
from scipy.stats import hmean

nltk.download("punkt_tab")

TEXT_METRIC_KEY = "texts"


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
            TEXT_METRIC_KEY: {
                "inputs": inputs,
                "labels": labels,
                "predictions": predictions,
            },
        }
    )

    return metrics


def load_nlp(lang: Literal["en", "fr"] = "en"):
    spacy_models = {"en": "en_core_web_md", "fr": "fr_core_news_md"}
    td_components = ["readability"]
    nlp = spacy.load(spacy_models[lang])
    for comp in td_components:
        nlp.add_pipe(f"textdescriptives/{comp}")
    return nlp


def compute_readability(texts, nlp):
    docs = nlp.pipe(texts)
    metrics_list = td.extract_dict(docs, include_text=False)
    merged_dict = reduce(lambda x, y: {**x, **y}, metrics_list)
    return {
        k: [merged_dict[k] for merged_dict in metrics_list] for k in merged_dict
    }


class FALCMetrics:
    def __init__(self, lang: Literal["en", "fr"] = "en") -> None:
        self.lang = lang
        self.nlp = load_nlp(self.lang)

        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("sacrebleu")
        self.sari = evaluate.load("sari")
        self.bertscore = evaluate.load("bertscore")
        self.readability = partial(compute_readability, nlp=self.nlp)

    def compute_bleu(self, predictions, references, **kwargs):
        # preprocessing
        preds = [pred.strip() for pred in predictions]
        refs = [[ref.strip()] for ref in references]

        result = self.bleu.compute(predictions=preds, references=refs)
        return {"bleu": result["score"]}

    def compute_rouge(self, predictions, references, **kwargs):
        def preprocess(texts):
            # rougeLSum expects newline after each sentence
            return [
                "\n".join(nltk.sent_tokenize(text.strip())) for text in texts
            ]

        preds = preprocess(predictions)
        refs = preprocess(references)

        result = self.rouge.compute(
            predictions=preds, references=refs, use_stemmer=True
        )
        return {k: round(v * 100, 4) for k, v in result.items()}

    def compute_sari(self, predictions, references, sources):
        preds = [pred.strip() for pred in predictions]
        refs = [[ref.strip()] for ref in references]
        srcs = [src.strip() for src in sources]

        result = self.sari.compute(
            predictions=preds, references=refs, sources=srcs
        )
        return result

    def compute_bertscore(self, predictions, references, **kwargs):
        preds = [pred.strip() for pred in predictions]
        refs = [ref.strip() for ref in references]

        def postprocess(result, rescale=False):
            return {
                f"bertscore_{k}{'_rescaled' if rescale else ''}": np.mean(
                    result[k]
                )
                * 100
                for k in ("f1", "recall", "precision")
            }

        def compute(rescale):
            return postprocess(
                self.bertscore.compute(
                    predictions=preds,
                    references=refs,
                    lang=self.lang,
                    rescale_with_baseline=rescale,
                ),
                rescale=rescale,
            )

        return {**compute(rescale=True), **compute(rescale=False)}

    def compute_readability(self, predictions, sources, **kwargs):
        preds = [pred.strip() for pred in predictions]
        srcs = [src.strip() for src in sources]

        def compression_ratio(ntok_src, ntok_tgt):
            return 100 - (ntok_tgt / ntok_src * 100)

        scores_preds = self.readability(preds)
        scores_sources = self.readability(srcs)

        return {
            "kmre": np.array(scores_preds["kandel_reading_ease"]).mean(),
            "lix": np.array(scores_preds["lix"]).mean(),
            "compression_ratio": compression_ratio(
                np.array(scores_sources["n_tokens"]),
                np.array(scores_preds["n_tokens"]),
            ).mean(),
        }

    def compute_novelty(self, predictions, sources, **kwargs):

        def unigram(text):
            text = self.nlp(textacy.preprocessing.remove.accents(text.lower()))
            return set(
                [
                    span.text
                    for span in textacy.extract.ngrams(
                        text, 1, filter_nums=True
                    )
                ]
            )

        def novelty(source, prediction):
            src_unigrams = unigram(source)
            pred_unigrams = unigram(prediction)
            try:
                return (
                    len(pred_unigrams - src_unigrams) / len(pred_unigrams) * 100
                )
            except:
                return 0.0

        return {
            "novelty": np.array(
                [novelty(s, p) for s, p in zip(sources, predictions)]
            ).mean()
        }

    def sari_rouge_bertscore_hmean(self, sari, rouge, bertscore):
        return {"sari_rouge_bertf1_hmean": hmean([sari, rouge, bertscore])}

    def compute(self, predictions, references, sources):
        kwargs = {
            "predictions": predictions,
            "references": references,
            "sources": sources,
        }

        res = {
            **self.compute_rouge(**kwargs),
            **self.compute_sari(**kwargs),
            **self.compute_bleu(**kwargs),
            **self.compute_bertscore(**kwargs),
            **self.compute_readability(**kwargs),
            **self.compute_novelty(**kwargs),
        }

        res.update(
            self.sari_rouge_bertscore_hmean(
                res["sari"], res["rougeL"], res["bertscore_f1"]
            )
        )

        return {k: round(v, 4) for k, v in res.items()}

    def __call__(self, predictions, references, sources):
        return self.compute(predictions, references, sources)
