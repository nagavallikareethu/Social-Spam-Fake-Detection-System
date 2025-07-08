# 📧📱📰 Social Spam & Fake Detection System

Welcome to **Smart Spam & Fake Content Detection** — a multi-model AI system that detects spam emails, SMS, fake news articles, and misleading social media posts.

---

## ✅ **Project Overview**

This project was developed as part of an internship to learn how to build, train, and deploy transformer-based NLP models in a real-world pipeline.

**What it does:**
- ✅ Detects if an **Email** is spam or ham (DistilBERT)
- ✅ Detects if an **SMS** is spam or ham (MobileBERT)
- ✅ Detects if a **News Article** is real or fake (BERT)
- ✅ Detects if **Social Media content** is real or fake (RoBERTa)

All in a **single, easy-to-use Streamlit app**!

---

## ⚙️ **Tech Stack**

| Layer | Technology |
|----------------|-----------------------------|
| 📂 Datasets | Kaggle: Emails, SMS, Fake/True News, Social Media |
| 🤖 Models | DistilBERT, MobileBERT, BERT, RoBERTa |
| 🧮 Preprocessing | Pandas, scikit-learn |
| ⚡ Fine-Tuning | PyTorch, Transformers |
| 🌐 Interface | Streamlit |
| 📦 Deployment | VS Code, GitHub |

---

## 🚀 **How to Run Locally**

1️⃣ **Clone this repo**

```bash
git clone https://github.com/nagavallikareethu/Social-Spam-Fake-Detection-System.git
cd Social-Spam-Fake-Detection-System
  

🗂️ Project Structure
bash
Copy
Edit
📂 fine_tuned_distilbert_email/   # Email spam model
📂 fine_tuned_mobilebert_sms/     # SMS spam model
📂 fine_tuned_bert_news/          # Fake/Real News model
📂 fine_tuned_roberta_social/     # Fake/Real Social Media model
📜 app.py                         # Streamlit Interface
📜 requirements.txt               # Dependencies
📜 README.md                      # This file!
