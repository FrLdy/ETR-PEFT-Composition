import unittest

from transformers import MBartConfig, MBartForConditionalGeneration

from .base import BaseRayTunerTest


class TestMBart(unittest.TestCase, BaseRayTunerTest):
    model_class = MBartForConditionalGeneration
    model_config = MBartConfig(
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        vocab_size=250027,
    )
    tokenizer_checkpoint = "facebook/mbart-large-50"
    training_config_special_kwargs = dict(
        is_causal_lm=False,
        tokenizer_kwargs=dict(src_lang="fr_XX", tgt_lang="fr_XX"),
    )
    data_config_special_kwargs = dict(
        tokenize_dataset=True,
    )
