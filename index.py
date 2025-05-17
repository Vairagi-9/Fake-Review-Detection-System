import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def preprocess_review(review):
 review = re.sub('[^a-zA-Z]', ' ', review)
 review = review.lower()
 review = review.split()
 ps = PorterStemmer()
 review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
 return ' '.join(review)
# Load your dataset
df = pd.read_csv('reviews.csv') # Ensure this path is correct# Preprocess reviews
df['processed_review'] = df['review'].apply(preprocess_review)
# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_review']).toarray()
y = df['label']
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Predicting a sample review
def predict_review(review_text):
 processed_review = preprocess_review(review_text)
 review_tfidf = tfidf.transform([processed_review]).toarray()
 return model.predict(review_tfidf)
# Example usage for prediction
review = "This product is absolutely amazing! I love it!"
prediction = predict_review(review)
print("Prediction (0 = Real, 1 = Fake):", prediction[0])