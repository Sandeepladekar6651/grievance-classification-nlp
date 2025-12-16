# Consumer Grievance Classification using NLP

## Project Overview
This project classifies consumer grievance complaints into predefined product categories
using Natural Language Processing (NLP) and Machine Learning.

## Input Features
- Issue
- Sub-issue
- Consumer complaint narrative

## Technologies Used
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

## Dataset
Source: Consumer Financial Protection Bureau (CFPB)  
Note: Dataset not included due to large size.

## Approach
1. Data cleaning and preprocessing
2. Text normalization (stopwords, punctuation removal)
3. TF-IDF feature extraction
4. Logistic Regression with class balancing
5. Model evaluation using accuracy and F1-score
6. Streamlit app for prediction

## How to Run
```bash
pip install -r requirements.txt
streamlit run griv.py
