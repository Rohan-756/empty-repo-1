import pickle
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("fake_review_ann_model.keras")

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Take input from user
review = input("Enter a review: ")

# Convert to TF-IDF
review_tfidf = tfidf.transform([review]).toarray()

# Predict
prediction = model.predict(review_tfidf)
predicted_class = (prediction > 0.5).astype("int32")

# Convert back to label
label = le.inverse_transform(predicted_class.flatten())

print("\nPrediction:", label[0])
