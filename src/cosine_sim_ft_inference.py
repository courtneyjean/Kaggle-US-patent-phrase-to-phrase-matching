'''
This process evaluates the effectiveness of predicting scores based on (rounded) cosine similarity of ft embeddings
'''

import config
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import stats
from sklearn import metrics
from create_word_embeddings import load_np_embeddings


def cal_cos_sim(em_1, em_2):
    cs = distance.cosine(em_1, em_2)
    cs = round(cs*4)/4  # round to the nearest 0.25 to match the training data score
    cs = int(cs*4)  # make into a integer to match preprocessing and make variable categorical
    return cs


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_PREPROCESSED)
    wv_anchor_embeddings, wv_target_embeddings = load_np_embeddings()

    print('length of anchor embeddings: {}'.format(wv_anchor_embeddings.shape))
    print('length of target embeddings: {}'.format(wv_target_embeddings.shape))

    cs_list = []
    for i in range(wv_anchor_embeddings.shape[0]):
        cs = cal_cos_sim(wv_anchor_embeddings[i], wv_target_embeddings[i])
        cs_list.append(cs)

    # calculate accuracy and pearson's correlation coff
    y_valid = df['score'].values.tolist()
    preds = cs_list

    accuracy = metrics.accuracy_score(y_valid, preds)
    pearsons_c, p_value = stats.pearsonr(y_valid, preds)

    print(f"Accuracy = {accuracy}")
    print(f"Pearsons Coefficient = {pearsons_c}")
    print(" ")
