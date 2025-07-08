import streamlit as st
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    MobileBertTokenizer, MobileBertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
import torch

# --------------------------------------
# ✅ Local paths to your fine-tuned models
# --------------------------------------
EMAIL_MODEL_DIR = r"C:\Users\nagav\Desktop\Spam_Detection\fine_tuned_distilbert_email\fine_tuned_distilbert_email"
SMS_MODEL_DIR = r"C:\Users\nagav\Desktop\Spam_Detection\fine_tuned_mobilebert_sms\fine_tuned_mobilebert_sms"
NEWS_MODEL_DIR = r"C:\Users\nagav\Desktop\Spam_Detection\fine_tuned_bert_news\fine_tuned_bert_news"
SOCIAL_MODEL_DIR = r"C:\Users\nagav\Desktop\Spam_Detection\roberta_social_model\fine_tuned_roberta_social"

# --------------------------------------
# ✅ Load DistilBERT for Email
# --------------------------------------
email_tokenizer = DistilBertTokenizer.from_pretrained(EMAIL_MODEL_DIR)
email_model = DistilBertForSequenceClassification.from_pretrained(EMAIL_MODEL_DIR)
email_model.eval()

# ✅ Load MobileBERT for SMS
sms_tokenizer = MobileBertTokenizer.from_pretrained(SMS_MODEL_DIR)
sms_model = MobileBertForSequenceClassification.from_pretrained(SMS_MODEL_DIR)
sms_model.eval()

# ✅ Load BERT for News Articles
news_tokenizer = BertTokenizer.from_pretrained(NEWS_MODEL_DIR)
news_model = BertForSequenceClassification.from_pretrained(NEWS_MODEL_DIR)
news_model.eval()

# ✅ Load RoBERTa for Social Media
social_tokenizer = RobertaTokenizer.from_pretrained(SOCIAL_MODEL_DIR)
social_model = RobertaForSequenceClassification.from_pretrained(SOCIAL_MODEL_DIR)
social_model.eval()

# --------------------------------------
# ✅ Streamlit UI
# --------------------------------------
st.set_page_config(
    page_title="Unified Content Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🧠 Multi-Content Detection System</h1>
    <p style='text-align: center;'>Check whether your Email, SMS, News Article, or Social Media Post is SPAM/FAKE or REAL using AI models.</p>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ✅ Select Model Type
st.subheader("Choose Content Type")
content_type = st.radio(
    "Select which type of text you want to check:",
    ("Email", "SMS", "News Article", "Social Media"),
    horizontal=True
)

# ✅ Text input
st.subheader(f"Enter your {content_type} text")
user_input = st.text_area("", placeholder="Type or paste here...", height=200)

# ✅ Predict Button
if st.button("🚀 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Analyzing... Please wait..."):
            if content_type == "Email":
                inputs = email_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = email_model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=1).item()
                result = "🚫 **SPAM**" if pred == 1 else "✅ **HAM**"
                st.success(f"📧 Email Prediction: {result}")

            elif content_type == "SMS":
                inputs = sms_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = sms_model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=1).item()
                result = "🚫 **SPAM**" if pred == 1 else "✅ **HAM**"
                st.success(f"📱 SMS Prediction: {result}")

            elif content_type == "News Article":
                inputs = news_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = news_model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=1).item()
                result = "✅ **REAL**" if pred == 0 else "🚫 **FAKE**"
                st.success(f"📰 News Prediction: {result}")

            elif content_type == "Social Media":
                inputs = social_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = social_model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=1).item()
                result = "✅ **REAL**" if pred == 0 else "🚫 **FAKE**"
                st.success(f"💬 Social Media Prediction: {result}")

# ✅ Footer
st.write("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px;'> | Internship Project |</p>",
    unsafe_allow_html=True
)
