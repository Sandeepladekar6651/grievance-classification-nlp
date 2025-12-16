import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

list_stopwords = stopwords.words('english')

def text_process(mess):
    # 1. remove punctuation
    nonpunc = [char for char in mess if char not in string.punctuation]
    nonpunc = "".join(nonpunc)
    # 2. remove stopwords
    return [word for word in nonpunc.split() if word not in list_stopwords]

# Load the saved model + transformer
with open("griv_log", "rb") as file:
    bow_transformer, logreg_model = pickle.load(file)   

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#2E86C1; font-size:42px;">📢 Smart Grievance Analyzer</h1>
    </div>
    """,
    unsafe_allow_html=True)

st.write("Enter complaint details below 👇")

# Input boxes
issue = st.text_input("Issue", "")
sub_issue = st.text_input("Sub-Issue", "")
complaint = st.text_area("Consumer Complaint Narrative", "")

# Predict Button
if st.button("Predict Category"):
    if issue.strip() == "" and sub_issue.strip() == "" and complaint.strip() == "":
        # Custom red warning
        st.markdown(
            "<span style='color:red; font-weight:bold;'>⚠️ Please enter some text.</span>",
            unsafe_allow_html=True
        )
    else:
        # Combine inputs
        user_text = issue + " " + sub_issue + " " + complaint

        # Transform and predict
        tdm = bow_transformer.transform([user_text])   
        prediction = logreg_model.predict(tdm)[0]

        # Map numeric labels back to category names
        category_map = {
            0: "Credit reporting",
            1: "Credit card / Prepaid card",
            2: "Debt collection",
            3: "Mortgage",
            4: "Checking / Savings account",
            5: "Money transfer / Virtual currency",
            6: "Payday loan / Personal loan",
            7: "Vehicle loan or lease",
            8: "Bank account or service",
            9: "Student loan",
            10: "Money transfers",
            11: "Credit reporting",
            12: "Debt or credit management",
            13: "Consumer Loan",
            14: "Other financial service",
            15: "Virtual currency"
        }

        # Get predicted category
        predicted_category = category_map.get(prediction, 'Unknown')

        # Show result in green bold text
        st.markdown(
            f"✅ Predicted Category: <span style='color:green; font-weight:bold;'>{predicted_category}</span>",

            unsafe_allow_html=True)
