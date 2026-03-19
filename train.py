import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# 1. Load Dataset
# Ensure you have run extract_csn_dataset.py first to get the 5000 samples
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("Error: dataset.csv not found. Run extract_csn_dataset.py first!")
    exit()

# 2. Preprocessing: EXACT match for Assignment Section 4 format
def combine_features(row):
    # Format: "Description Parameters return ReturnType keywords Keywords"
    desc = str(row['description'])
    params = str(row['parameters'])
    ret = str(row['return_type'])
    keys = str(row['keywords'])
    return f"{desc} {params} return {ret} keywords {keys}"

df['combined_text'] = df.apply(combine_features, axis=1)

X = df['combined_text']
y = df['function_name']

# 3. Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Build a Lightweight Pipeline
# We remove standard English stop_words because 'return' is a stop word 
# but it's a vital marker for our model structure.
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), 
    ('clf', MLPClassifier(
        hidden_layer_sizes=(100,), 
        max_iter=500, 
        early_stopping=True, # Prevents overfitting
        random_state=42
    ))
])

# 5. Train
print(f"Training on {len(df)} samples...")
model_pipeline.fit(X, y_encoded)

# 6. Save assets
joblib.dump(model_pipeline, 'model.pkl')
joblib.dump(label_encoder, 'encoder.pkl')
print("Model and Encoder saved successfully.")