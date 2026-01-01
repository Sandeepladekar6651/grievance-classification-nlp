# Consumer Grievance Classification using NLP

## 📌 Project Overview
This project classifies consumer grievance complaints into predefined financial
product categories using **Natural Language Processing (NLP)** and
**Machine Learning**.  
The solution is deployed as an interactive **Streamlit web application**.

---

## 🧾 Input Features
- Issue
- Sub-issue
- Consumer complaint narrative

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes
- Streamlit

---

## 📊 Dataset
**Source:** Consumer Financial Protection Bureau (CFPB)  
**Note:** Dataset is not included due to large size.

---

## ⚙️ Approach
1. Data cleaning and preprocessing
2. Text normalization (stopword removal, punctuation removal)
3. Feature extraction using **CountVectorizer**
4. Model training using **Multinomial Naive Bayes**
5. Model evaluation using accuracy and F1-score
6. Model serialization using pickle
7. Deployment as a Streamlit web application

---

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run griv.py
