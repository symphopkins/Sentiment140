# Sentiment140 Models

## Overview

Optimized data processing through Python pipelines and applied advanced natural language processing techniques to develop several sentiment classification models. Conducted a comparative analysis, leveraging accuracy metrics from classification reports to identify the most effective model for classifying tweet polarity.

## Files Included
- `Sentiment140.py`: Python script containing the code.
- `Sentiment140.ipynb`: Google Colab Notebook containing the detailed code implementation and explanations.
- `requirements.txt`: Text file listing the required Python packages and their versions.
- `LICENSE.txt`: Text file containing the license information for the project.


## Installation
To run this project, ensure you have Python installed on your system. You can install the required dependencies using the `requirements.txt` file.

### Usage
To utilize the provided code for your own sentiment analysis tasks, follow these steps:
1. Ensure you have the dataset (sentiment140.csv) accessible. Use Pandas to read the dataset, select relevant columns, and clean the text data.
2. Use the code provided to build and evaluate different models. You can start with the initial Support Vector Classifier (SVC) model and then explore other classifiers like Logistic Regression and Random Forest Classifier using GridSearchCV for hyperparameter optimization.
3. Evaluate the models using confusion matrices, classification reports, and visualizations provided in the code.

## Data Source
The dataset used for this project is sentiment140.csv sourced from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Citations
Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

## License
MIT License
