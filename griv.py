import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords (required for Streamlit Cloud)
nltk.download('stopwords')

stop_words = stopwords.words('english')

def text_process(text):
    """
    Remove punctuation and stopwords
    """
    text = "".join([c for c in text if c not in string.punctuation])
    return [word for word in text.split() if word.lower() not in stop_words]

# Load trained model + vectorizer (Naive Bayes)
with open("models/grievance_model.pkl", "rb") as f:
    bow_transformer, model = pickle.load(f)

# App title
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#2E86C1; font-size:42px;">📢 Smart Grievance Analyzer</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Enter complaint details below 👇")

# Input fields
issue = st.text_input("Issue")
sub_issue = st.text_input("Sub-Issue")
complaint = st.text_area("Consumer Complaint Narrative")

# Predict button
if st.button("Predict Category"):
    if issue.strip() == "" and sub_issue.strip() == "" and complaint.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Combine user inputs
        user_text = issue + " " + sub_issue + " " + complaint

        # Transform input
        tdm = bow_transformer.transform([user_text])

        # Predict category
        prediction = model.predict(tdm)[0]

        # Confidence score
        proba = model.predict_proba(tdm)
        confidence = proba.max() * 100  # percentage

        # Category mapping
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

        # Display results
        st.success(f"✅ Predicted Category: **{category_map[prediction]}**")
        st.info(f"📊 Confidence Score: **{confidence:.2f}%**")
