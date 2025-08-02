# model_builder.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset from a public URL
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
df = pd.read_csv(url)

# Prepare the data
# Features (X) are the measurements of the flowers
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
# Target (y) is the species of the flower
y = df['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
# n_estimators is the number of trees in the forest
# random_state ensures reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
# This file will be loaded by our Streamlit app
joblib.dump(model, 'iris_model.joblib')

print("Model trained and saved as iris_model.joblib")