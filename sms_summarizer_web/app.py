# app.py
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from collections import Counter
import re

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert_sms_final")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert_sms_final",
        use_safetensors=True
    )
    id2label = joblib.load("bert_sms_final/id2label.pkl")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, id2label, device

tokenizer, model, id2label, device = load_model()

def clean_text(t): 
    return re.sub(r'[^a-z0-9\s%₹]', '', str(t).lower()).strip()

def predict(t):
    if not t.strip(): return "others"
    enc = tokenizer(clean_text(t), return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(**enc).logits, dim=1).item()
    return id2label[pred].lower()  # ← LOWERCASE

st.title("Intelligent SMS Summarizer")
uploaded = st.file_uploader("Upload SMS CSV (text column)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    if 'text' not in df.columns:
        st.error("Need 'text' column")
    else:
        df['clean'] = df['text'].apply(clean_text)
        df = df[df['clean'] != ''].copy()
        with st.spinner("Classifying..."):
            df['label'] = df['clean'].apply(predict)
        
        # FIX: LOWERCASE LABELS
        promo_labels = ['telecom', 'shopping', 'real estate', 'banking', 'marketing']
        promo = df[df['label'].isin(promo_labels)]
        spam = df[df['label'] == 'spam']
        
        counts = Counter(promo['label'])
        parts = []
        for cat in promo_labels:
            if counts[cat]: 
                pretty = cat.replace("real estate", "real-estate")
                parts.append(f"{counts[cat]} {pretty} offers")
        summary = ", ".join(parts).capitalize() + "." if parts else "No offers found."
        if len(spam): summary += f"\nWarning: {len(spam)} SPAM filtered."
        
        st.success("Done!")
        st.markdown(f"### {summary}")
        
        if len(promo) > 0:
            st.write("### Top Offers:")
            for cat in promo_labels:
                samples = promo[promo['label'] == cat]['text'].head(2)
                for s in samples:
                    st.write(f"• {s}")
        else:
            st.warning("No promotional messages — check model training")
        
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "classified.csv", "text/csv")