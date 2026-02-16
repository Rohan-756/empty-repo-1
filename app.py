import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/v2/fake_review_ann_model.keras")
    with open("models/v2/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/v2/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tfidf, le

model, tfidf, le = load_model()

st.title("🕵️ Fake Review Detection System")
st.write("Enter a product review to check whether it is Genuine or Fake.")

# User input
review = st.text_area("Enter Review Here:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_tfidf = tfidf.transform([review]).toarray()
        prediction = model.predict(review_tfidf)
        predicted_class = (prediction > 0.5).astype("int32")
        label = le.inverse_transform(predicted_class.flatten())

        if label[0] == "CG":
            st.error("🚨 This Review is FAKE")
        else:
            st.success("✅ This Review is GENUINE")
