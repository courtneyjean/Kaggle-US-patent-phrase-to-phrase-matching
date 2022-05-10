import config
import model_dispatcher
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from scipy import stats
from sklearn import linear_model, tree
from xgboost import XGBClassifier
from sklearn import metrics
from create_word_embeddings import load_np_embeddings


def create_corpus(df):
    corpus = []
    for col in df.columns.values.tolist():
        corpus = corpus + df[col].values.tolist()
    return corpus


def create_vocabulary(corpus):
    ctv = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
    ctv.fit(corpus)
    return ctv


def run(model_name, preproc):
    # read in pre-processed file
    fn = config.TRAINING_PREPROCESSED
    df = pd.read_csv(fn)

    if preproc == 'bow':
        ctv = create_vocabulary(df['input_text'].tolist())

    # Create a numpy to store the results
    np_headings = ['fold', 'accuracy', 'pearsons']
    np_scores = np.zeros([5, 3])

    for fold in range(config.K_SPLITS):
        df_train = df[df.kfold != fold].reset_index(drop=True)  # use only data not equal to provided fold
        df_valid = df[df.kfold == fold].reset_index(drop=True)  # validation data is equal to provided fold

        df_train_idx = df.index[df.kfold != fold]
        df_valid_idx = df.index[df.kfold == fold]

        if preproc == 'ft':
            wv_input_embeddings = load_np_embeddings()
            x_train = wv_input_embeddings[df_train_idx]
            x_valid = wv_input_embeddings[df_valid_idx]

        if preproc == 'bow':
            x_train = ctv.transform(df_train['input_text']).toarray()
            x_valid = ctv.transform(df_valid['input_text']).toarray()

        # get the labels
        y_train = df_train['score'].tolist()
        y_valid = df_valid['score'].tolist()

        model = model_dispatcher.models[model_name]

        model.fit(x_train, y_train)
        preds = model.predict(x_valid).tolist()


        accuracy = metrics.accuracy_score(y_valid, preds)
        pearsons_c, p_value = stats.pearsonr(y_valid, preds)

        # print(f"Fold: {fold}")
        # print(f"Accuracy = {accuracy}")
        # print(f"Pearsons Coefficient = {pearsons_c}")
        # print(" ")

        # Store results in numpy
        np_scores[fold, 0] = round(fold, 4)
        np_scores[fold, 1] = round(accuracy, 4)
        np_scores[fold, 2] = round(pearsons_c, 4)

    return pd.DataFrame(np_scores, columns=np_headings), np_scores.mean(axis=0)[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--preproc", type=str)
    args = parser.parse_args()

    np_scores_df, mean_scores = run(model_name=args.model, preproc=args.preproc)

    print("Evaluation complete for {}".format(args.model))
    print(np_scores_df)
    print(f"Mean Accuracy: {mean_scores[0]}, Mean Pearsons Coefficient {mean_scores[1]}")


