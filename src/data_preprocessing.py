"""
These functions are used for preprocessing the training and test data.
"""

import re
import string
import config
import pandas as pd
from sklearn import model_selection
from nltk.stem.snowball import SnowballStemmer


def clean_text_strings(s):
    """
    :param s: a text string to be cleaned
    :return: a clean text string
    """
    # Remove punctuation
    s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)

    # make lower case
    s = s.lower()

    # remove any weird spacing issues
    s = s.split()
    s = " ".join(s)

    return s


def make_target_categorical(score):
    """
    The score column is currently scored as 0-1.  This converts it to 0-4
    :param score: a single score from the target column (score) of the dataset
    :return: an int representing a category between 0-4
    """
    return int(score*4)


def add_context_desc(df):
    # break out section and group from context code
    df["section"] = df["context"].str[:1]
    df["group"] = df["context"].str[1:3]

    # read in look up tables
    section_df = pd.read_csv('../input/section_lookup_tb.csv')
    group_df = pd.read_csv('../input/group_lookup_tb.csv')

    # merge tables
    df = pd.merge(df, section_df, how="left", left_on='section', right_on='section_code')
    df = pd.merge(df, group_df, how="left", left_on='context', right_on='group_code')

    df['context_text'] = df['section_text'] + " " + df['group_text']

    # drop unnecessary columns from table
    df = df.drop(['section', 'group', 'group_text', 'section_text', 'group_code', 'section_code'], axis=1)

    return df


def create_combined_context_text_strings(df):
    '''
    # combine the text and context
    df['anchor_w_context'] = df['anchor'] + " " + df['context_text']
    df['target_w_context'] = df['target'] + " " + df['context_text']

    # preprocess the string columns
    df['anchor_w_context'] = df['anchor_w_context'].apply(lambda x: clean_text_strings(x))
    df['target_w_context'] = df['target_w_context'].apply(lambda x: clean_text_strings(x))
    '''

    df['input_text'] = df['context_text'] + config.SEP + df['anchor'] + config.SEP + df['target']
    df['input_text'] = df['input_text'].apply(lambda x: clean_text_strings(x))

    return df


def create_folds(df, target_col_name, ns):
    df["kfold"] = -1  # create new column
    df = df.sample(frac=1).reset_index(drop=True)  # randomise rows

    # The targets are floats, which sklearn interprets as continuous.
    # I am changing them to be strings, so they are interpreted as multiclass.
    df[target_col_name] = df[target_col_name].astype(str)

    y = df[target_col_name].values  # fetch targets
    kf = model_selection.StratifiedKFold(n_splits=ns)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    # convert targets back to float
    df[target_col_name] = df[target_col_name].astype(float)

    return df


if __name__ == "__main__":

    # Will probably actually call this on a folds training set from inside training
    df = pd.read_csv(config.TRAINING_FILE)

    # make target categorical
    df[config.TARGET_NAME] = df[config.TARGET_NAME].apply(lambda x: make_target_categorical(x))

    # Create k fold splits
    target_col_name = config.TARGET_NAME
    ns = config.K_SPLITS
    df = create_folds(df, target_col_name, ns)

    # Add context description to file
    df = add_context_desc(df)

    # create the combine context string
    df = create_combined_context_text_strings(df)

    # apply stemming to the anchor_w_context and the target_w_context strings
    # stemmer = SnowballStemmer("english")
    # df['input_text'] = df['input_text'].apply(lambda x: stemmer.stem(x))
    # df['anchor_w_context'] = df['anchor_w_context'].apply(lambda x: stemmer.stem(x))
    # df['target_w_context'] = df['target_w_context'].apply(lambda x: stemmer.stem(x))

    df.to_csv('../input/train_preprocessed.csv', index=False)
