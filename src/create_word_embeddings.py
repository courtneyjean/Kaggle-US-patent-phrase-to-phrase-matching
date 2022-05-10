'''
This script applies word embeddings form FastText
'''

import config
import numpy as np
import pandas as pd
import fasttext


def load_np_embeddings():
    # modify the default parameters of np.load
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    # wv_anchor_embeddings = np.load(config.FT_EMBEDDINGS_ANCHOR)
    # wv_target_embeddings = np.load(config.FT_EMBEDDINGS_TARGET)

    wv_input_embeddings = np.load(config.FT_EMBEDDINGS_INPUT)

    # restore np.load for future normal usage
    np.load = np_load_old

    # return wv_anchor_embeddings, wv_target_embeddings
    return wv_input_embeddings

if __name__ == "__main__":
    # Get preprocessed data
    df = pd.read_csv(config.TRAINING_PREPROCESSED)

    # load fast text model
    ft_model = fasttext.load_model(config.PRETRAINED_FT_EMBEDDINGS_MODEL)

    # create embeddings for all sentences - first attempt for cosine sim
    #  wv_anchor_embeddings = np.array(df['anchor_w_context'].apply(lambda x: ft_model.get_sentence_vector(x)))
    #  wv_target_embedding = np.array(df['target_w_context'].apply(lambda x: ft_model.get_sentence_vector(x)))

    ''' second attempt
    wv_anchor_embeddings_l = []
    for txt in df['anchor_w_context']:
        wv_anchor_embeddings_l.append(ft_model.get_sentence_vector(txt))
    wv_anchor_embeddings_np = np.array(wv_anchor_embeddings_l)

    wv_target_embeddings_l = []
    for txt in df['anchor_w_context']:
        wv_target_embeddings_l.append(ft_model.get_sentence_vector(txt))
    wv_target_embeddings_np = np.array(wv_target_embeddings_l)
    '''

    wv_input_embeddings_l = []
    for txt in df['input_text']:
        wv_input_embeddings_l.append(ft_model.get_sentence_vector(txt))
    wv_input_embeddings_np = np.array(wv_input_embeddings_l)

    # np.save(config.FT_EMBEDDINGS_ANCHOR, wv_anchor_embeddings_np)
    # np.save(config.FT_EMBEDDINGS_TARGET, wv_target_embeddings_np)
    np.save(config.FT_EMBEDDINGS_INPUT, wv_input_embeddings_np)
