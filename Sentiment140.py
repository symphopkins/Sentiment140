# -*- coding: utf-8 -*-

# importing library
import pandas as pd

# reading the file; we had to change the encoding from utf-8 to ISO-8859-1 because there were issues with encoding
sentiment140_df = pd.read_csv(r"sentiment140.csv", encoding='ISO-8859-1', header=None)
sentiment140_df.head()


# displaying shape
sentiment140_df.shape

# selecting relevant columns
sentiment140_df = sentiment140_df[[0,5]]

# renaming columns
sentiment140_df = sentiment140_df.rename(columns={0:'polarity', 5: 'tweet'})
sentiment140_df.head()

# displaying info
sentiment140_df.info()

# checking for missing data
sentiment140_df.isnull().sum() * 100 / len(sentiment140_df)

# checking the target distribution
print(sentiment140_df['polarity'].value_counts(normalize=True))


# creating two separate dataframes based on the polarity of the tweets
sentiment140_neg = sentiment140_df.loc[sentiment140_df['polarity']== 0]
sentiment140_pos = sentiment140_df.loc[sentiment140_df['polarity']== 4]

# sampling from the dataframes
sentiment140_neg = sentiment140_neg.sample(n=1000, random_state=42)
sentiment140_pos = sentiment140_pos.sample(n=1000, random_state=42)

# concatenating the dataframes
sentiment140_df = pd.concat([sentiment140_neg, sentiment140_pos], ignore_index=True)

# checking the target distribution
print(sentiment140_df['polarity'].value_counts(normalize=True))

# importing libraries
import re
from sklearn.base import TransformerMixin

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # creating our transformer to clean the texts
# class features(TransformerMixin):
#     def transform(self, X, **transform_params):
#         """Override the transform method to clean text"""
#         return [clean_text(text) for text in X]
# 
#     def fit(self, X, y= None, **fit_params):
#         return self
# 
#     def get_params(self, deep= True):
#         return {}
# 
# # defining function to clean the text
# def clean_text(text):
#     """Removing nan, @airline, punctuation, URL, or any non alpanumeric characters and converting the text into lowercase"""
#     # Remove nan, @airline, punctuation, URL, or any non alpanumeric characters and seperate word using a single space.
#     text = ' '.join(re.sub("(nan)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
#     # remove all the leading and trailing spaces from a string and convert the text into lowercase
#     return text.strip().lower()
# 
# # applying the function to each row of the tweet column
# sentiment140_df['tweet'] = sentiment140_df['tweet'].apply(clean_text)
# 
# # displaying the results
# sentiment140_df.head()

# importing library
from sklearn.model_selection import train_test_split

# creating feature and label variables
X = sentiment140_df['tweet']
y = sentiment140_df['polarity']

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
print(f'X_train dimension: {X_train.shape}; y_train dimension: {y_train.shape}')
print(f'X_test dimension: {X_test.shape}; y_train dimension: {y_test.shape}')

# importing libraries
import string
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

# loading the small model
nlp = spacy.load("en_core_web_sm")

stop_words = spacy.lang.en.stop_words.STOP_WORDS

# creating a tokenzer function from a given sentence
def spacy_tokenizer(sentence):

    # Split the sentence into tokens/words
    mytokens = nlp(sentence)
    # Removing stop words and obtain the lemma
    mytokens = [ word.lemma_ for word in mytokens if word not in stop_words]
    return mytokens

# importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from time import time

# timing our codes
t0 = time()

# overriding the string tokenization step while preserving the preprocessing and n-grams generation steps
# since we are using a customized tokenizer, the token_pattern parameter will be set to None
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1,1), token_pattern=None)

# creating a support vector classifier
classifier = SVC()

# creating a pipeline based on the cleaner, vectorizer and clasifier
pipeline = Pipeline ([("cleaner", features()),
                 ("vectorizer", tfidf_vector),
                 ("classifier", classifier)])

# fitting the model
pipeline.fit(X_train, y_train)

# displaying codes' time
print(f"It takes about {time() - t0:.1f} seconds")

# importing libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# specifying the target names since the y is label encodes using 0 and 4
target_names = ['negative', 'positive']

# plotting non-normalized confusion matrix and normalized confusion matrix
titles_options = [("Confusion Matrix Without Normalization", None),
                  ("Normalized Confusion Matrix", 'true')]
# looping over the two cases of confusion matrix
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test,
                                 display_labels= target_names,
                                 cmap=plt.cm.Blues, # matplotlib Colormap
                                 # normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population
                                 normalize=normalize)
    print(title)

plt.show()

# creating classification report
# predicting the test data
y_pred = pipeline.predict(X_test)

# printing out the report
print(classification_report(y_test, y_pred, target_names = target_names))


# importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# creating pipeline using cleaner, tf-idf and SVC
pipeline = Pipeline ([("cleaner", features()),
                 ("vectorizer", TfidfVectorizer(tokenizer = spacy_tokenizer, token_pattern=None)),
                 ("classifier", SVC())])

# specifying the hyperparameters using dictionary based on the following format
# estimator/transformerName__(double understore) corresponding parameter : potential parameter
parameters = {
    'vectorizer__max_df': (0.5, 1.0),
    'vectorizer__ngram_range': ((1, 1), (1, 2)), #unigrams or bigrams
    'vectorizer__use_idf': (True, False),
    'classifier__kernel': ['linear', 'rbf']
}

# performing gridsearch CV
grid_search = GridSearchCV(pipeline, parameters, n_jobs=None, verbose=1)
print("Performing grid search...")
print("The pipeline contains:", [name for name, _ in pipeline.steps])
print("parameters are as follows:")
pprint(parameters)

# timing the grid search
t0 = time()
grid_search.fit(X_train, y_train)
print(f"It takes about {time() - t0:.1f} seconds.")
print()

# printing the best score and parameters
print(f"Best score= {grid_search.best_score_:0.3f}")
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    # https://stackoverflow.com/questions/2354329/what-is-the-meaning-of-r
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# plotting non-normalized confusion matrix and normalized confusion matrix
titles_options = [("Confusion Matrix Without Normalization", None),
                  ("Normalized Confusion Matrix", 'true')]
# looping over the two cases of confusion matrix
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test,
                                 display_labels= target_names,
                                 cmap=plt.cm.Blues, # matplotlib Colormap
                                 # normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population
                                 normalize=normalize)
    print(title)

plt.show()

# creating classification report
# predicting the test data
y_pred = grid_search.predict(X_test)

# printing out the report
print(classification_report(y_test, y_pred, target_names = target_names))


# importing libraries
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# defining the estimator class to handle different classifiers
class ClfSwitcher(BaseEstimator):
    # initializing the estimator
    def __init__(
        self,
        estimator = None,
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        estimator is a class variable/attribute/property;
        It is a machine learning algorithm/estimator.
        You can find all available  classification estimators in sklearn at
          https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
        """

        self.estimator = estimator

    # fitting the model using the given estimator
    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    # predicting the label using the given estimator
    def predict(self, X, y=None):
        return self.estimator.predict(X)

    # predicting the probability using the given estimator
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    # computing the score using the given estimator
    def score(self, X, y):
        return self.estimator.score(X, y)

"""Now we can tune and fit the model."""

# creating the pipeline of cleaner, tfidf and classifier to be specified by ClfSwitcher
# token_pattern=None because we have a custom tokenizer
pipeline = Pipeline ([("cleaner", features()),
                 ("vectorizer", TfidfVectorizer(tokenizer = spacy_tokenizer, token_pattern=None)),
                 ("classifier", ClfSwitcher())])

# creating the hyperparameters using the following format
# For the cleaner and vectorizer:
# Estimator/transformer name__(double understore) corresponding parameter : potential parameter
# The classifier handling different classifiers has three levels
# classifier__estimator__corresponding parameter : potential parameter
parameters = [
    {
        'vectorizer__max_df': (0.5, 1.0),
        'vectorizer__ngram_range': ((1, 1), (1, 2)), #unigrams or bigrams
        'vectorizer__use_idf': (True, False),
        'classifier__estimator': [SVC()],
    },
    {
        'vectorizer__max_df': (0.5, 1.0),
        'vectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'vectorizer__use_idf': (True, False),
        'classifier__estimator': [LogisticRegression()],
    },
    {
        'vectorizer__max_df': (0.5, 1.0),
        'vectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'vectorizer__use_idf': (True, False),
        'classifier__estimator': [RandomForestClassifier()]
    },
]

# performing the grid search
print("Performing grid search...")
print("The pipeline contains:", [name for name, _ in pipeline.steps])
print("parameters are as follows:")
pprint(parameters)

# timing grid search
t0 = time()
gscv = GridSearchCV(pipeline, parameters, cv=5, n_jobs= None, return_train_score=False, verbose=3)
gscv.fit(X_train, y_train)
print(f"It takes about {time() - t0:.3f} seconds")

# printing the best parameters and score
print(f"Best score= {gscv.best_score_:0.3f}")
best_parameters = gscv.best_estimator_.get_params()

# looping over the parameters and get all potential algorithms in the pipeline
all_classifiers =[]
for parameter in parameters:
    all_classifiers.append(parameter['classifier__estimator'])
all_classifiers = [str(alg) for clf in all_classifiers for alg in clf]
print("All potential classifiers:")
pprint(all_classifiers)

# finding the location/index of the optimal classifier
idx = all_classifiers.index(str(best_parameters['classifier__estimator']))
print("Best parameters set:")
for param_name in sorted(parameters[idx].keys()):
    # https://stackoverflow.com/questions/2354329/what-is-the-meaning-of-r
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# importing libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# predicting the labels of the test dataset
y_pred = gscv.predict(X_test)
# creating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# plotting the heatmap of the confusion matrix without normalization
ax = plt.axes()
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(cm, annot=True, fmt = 'd', xticklabels = target_names, yticklabels = target_names, cmap="YlGnBu")
ax.set_title("Confusion matrix, without normalization")
plt.show()

# normalizing the confusion matrix
cm = cm / cm.astype(np.float64).sum(axis=1)[:,None]
# plotting the heatmap of the normalized confusion matrix
ax = plt.axes()
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(cm, annot=True, fmt = '.2f', xticklabels = target_names, yticklabels = target_names, cmap="YlGnBu")
ax.set_title("Normalized confusion matrix")
plt.show()

# displaying the classification report
print(classification_report(y_test, y_pred, target_names = target_names))
