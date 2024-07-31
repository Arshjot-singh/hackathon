import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Step 1: Load the CSV file
file_path = r'D:\Desktop\BlackRock\synthetic_green_bond_data_with_real_issuers (2).csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
st.write("Dataset Preview:")
st.write(data.head())

# Step 2: Handling Missing Values
data = data.dropna()

# Step 3: Drop specified columns
columns_to_drop = ['Issuer_Location', 'Issuer_Sector', 'Credit_Rating']
data_dropped = data.drop(columns=columns_to_drop)

# Step 4: Label Encode 'Risk_Involved' Feature
label_encoder = LabelEncoder()
data_dropped['Risk_Involved'] = label_encoder.fit_transform(data_dropped['Risk_Involved'])

# Step 5: Prepare the data for modeling
X = data_dropped.drop('Risk_Involved', axis=1)
y = data_dropped['Risk_Involved']
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Issuer_Name']), y, test_size=0.2, random_state=42)

# Step 6: Train the RandomForestClassifier model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 7: Make predictions and evaluate the model
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.4f}")

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix)

