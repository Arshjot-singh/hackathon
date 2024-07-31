import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

# Load the CSV file
file_path = r'D:\Desktop\BlackRock\synthetic_green_bond_data_with_real_issuers (2).csv'
data = pd.read_csv(file_path)

# Handle Missing Values
data = data.dropna()

# Drop specified columns
columns_to_drop = ['Issuer_Location', 'Issuer_Sector', 'Credit_Rating']
data_dropped = data.drop(columns=columns_to_drop)

# Label Encode 'Risk_Involved' Feature
label_encoder = LabelEncoder()
data_dropped['Risk_Involved'] = label_encoder.fit_transform(
    data_dropped['Risk_Involved'])

# Prepare the data for modeling
X = data_dropped.drop('Risk_Involved', axis=1)
y = data_dropped['Risk_Involved']
X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['Issuer_Name']), y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the model and other necessary data
model_data = {
    'model': rf_classifier,
    'label_encoder': label_encoder,
    'features': X.drop(columns=['Issuer_Name']).columns.tolist()
}

with open('green_bond_risk_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

# Function to predict risk for a given bond name


def predict_risk_for_bond(bond_name):
    bond_data = data_dropped[data_dropped['Issuer_Name'] == bond_name]
    if bond_data.empty:
        return None
    bond_features = bond_data.drop(columns=['Risk_Involved', 'Issuer_Name'])
    with open('green_bond_risk_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    predicted_risk = model.predict(bond_features)
    predicted_risk_category = label_encoder.inverse_transform(predicted_risk)
    return predicted_risk_category[0]


# Streamlit interface for user input and prediction
st.title("Bond Risk Prediction App")
bond_name = st.text_input("Enter the bond name:")
if bond_name:
    predicted_risk = predict_risk_for_bond(bond_name)
    if predicted_risk is not None:
        st.write(
            f"The predicted risk for the bond '{bond_name}' is: {predicted_risk}")
    else:
        st.write("Bond name not found in the dataset.")
