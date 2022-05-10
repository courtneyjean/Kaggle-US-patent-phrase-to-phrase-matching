# Kaggle Comp: U.S. Patent Phrase to Phrase Matching: Evaluating approaches for natural language text similarity

May 2022

## INTRODUCTION
The recent Kaggle competition “U.S. Patent Phrase to Phrase Matching” asks competitors to train models to match key phrases in patient documents and score the semantic similarity of the text.  Whilst the competition winners will likely use state of the art and computational heavy transformers, I wanted to use the competition to apply some common NLP approaches and evaluate their relative performance. 

More information on this competition can be found here: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview

## SETUP

I've utilised the file structure suggested in "Approaching (almost) any machine learning problem" by Abhishek Thakur, which provides an intuitive project structure for building and evaluating machine learning models.

A virtual environment for running this project is included in the environment.yml file.  

To create this environment, run: conda env create -f environment.yml
To activate this environment, run: conda activate phrase_match

## TARGET

In this competition, the aim is to predict the similarity score between two phrases.  I will treat this as a classification
problem, with the potential options being the following:

1.0 - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. “the”, “and”, “or”).
0.75 - Close synonym, e.g. “mobile phone” vs. “cellphone”. This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
0.5 - Synonyms which don’t have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
0.25 - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
0.0 - Unrelated.

## MODEL EVALUATION 

The model will be evaluated using a stratified k fold cross validation approach, to ensure that an equal number of each of the 
classification scores above is represented in each fold.  The stratified fold approach is created in src/create_folds.py

The competition metric used to evaluate submissions is pearson's coefficient, which outputs a single number between -1 and
1 to indicate the similarity between the predicted and actual scores. I'm calculating this using the scipy implementations of 
pearson's coefficient.

## PREPROCESSING

CONTEXT: The competition data includes the Cooperative Patent Classification (CPC) code, and advises that the similarity 
of the text should be considered within the patients "contex".  To ensure that the text similarity methods I am using also
consider the semantic similarity of the context code, I'm replacing the code with its definition.  To do this, I've created 
two lookup tables that I've added to the input folder.  Code to create these (and more info on source), 
is here: notebooks/EDA_context_CPC_Codes.ipynb.  In the transform variables script, I've imported the descriptions into 
the train file.

INPUTS: I've settled on the approach of simplifying the input text to be in the format: context_text + anchor_text + target

TARGET: I'm approaching the problem as a multi-class classification, where the aim is to predict the score as one of 
5 categories (see target).  To support this, I'm converting the target column from scores 0-1, to 0-4.




















