import numpy as np


def replace_ignore_index(tokens, substitution_token, ignore_index=-100):
    tokens = np.where(tokens != ignore_index, tokens, substitution_token)
    return tokens
