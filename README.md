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

## PREPROCESSING

CONTEXT: The competition data includes the Cooperative Patent Classification (CPC) code, and advises that the similarity 
of the text should be considered within the patients "contex".  To ensure that the text similarity methods I am using also
consider the semantic similarity of the context code, I'm replacing the code with its definition.  To do this, I've created 
two lookup tables that I've added to the input folder.  Code to create these (and more info on source), 
is here: notebooks/EDA_context_CPC_Codes.ipynb.  In the transform variables script, I've imported the descriptions into 
the train file.

INPUTS: I experimented with two approaches:
1) Two input vectors of context_text + target and anchor_text + target, joined horiziontally.  
2) Single input text vector to be in the following format: context_text + anchor_text + target.  

## TOKENISATION AND EMBEDDINGS
Machine learning algorthims require natural language (ie text) to be converted to numerical values.  I've experimented with two different approaches:
1) Bag of Words approach using Sklearn's CountVectorisation method
2) Pretrained FastText word embeddings: a pretrained model that produces vectors representing the n-grams present in the text 

## MODEL EVALUATION 

The model will be evaluated using a stratified k fold cross validation approach, to ensure that an equal number of each of the 
classification scores above is represented in each fold.  The stratified fold approach is created in src/create_folds.py

The competition metric used to evaluate submissions is pearson's coefficient, which outputs a single number between -1 and
1 to indicate the similarity between the predicted and actual scores. I'm calculating this using the scipy implementations of 
pearson's coefficient.

### Comparing default implementations of simple models

To begin with, I ran a couple of experiments to gather a performance baseline for the various preprocessing options and the default implementations of different algorthimic approaches.

| Preprocessing | Tokenisation | Model | Pearson's Coefficient |
|--|--|--|--:|
|Two input vectors| Bag of Words | Logistic Regression | 0.2531|
|Two input vectors| Bag of Words | Decision Tree | 0.3464|
|Two input vectors| Bag of Words | XGBoost| 0.2403|
|Two input vectors + stemming| FastText | Logistic Regression | 0.0175|
|Two input vectors + stemming| FastText | Decision Tree | 0.0132|
|Two input vectors + stemming| FastText | XGBoost| 0.0184|
|Single input vector| FastText | Logistic Regression | 0.3015|
|Single input vector| FastText | Decision Tree | 0.1101|
|Single input vector| FastText | XGBoost| 0.1450|
|Single input vector| Bag of Words | Decision Tree | 0.3303|

From this I could see that the stemming didn't appear to be helping the preprocessing, so I removed it from preprocessing.  The tree models (DecisionTree and XGBoost) appeared to be doing best, and within the performance of these models, there wasn't much difference between using teh bag of words approach, or the FastText approach.  I decided to proceed with the FastText approach, as it was quicker.

