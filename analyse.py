import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and data
@st.cache_resource
def load_model():
    with open('green_bond_risk_model.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_data
def load_data():
    return pd.read_csv(r'D:\Desktop\BlackRock\synthetic_green_bond_data_with_real_issuers (2).csv')

model_data = load_model()
data = load_data()

# Set up the Streamlit app
st.title('Green Bond Risk Predictor')

# User input
bond_name = st.text_input('Enter the bond name:')

if bond_name:
    # Find the bond in the dataset
    bond_data = data[data['Issuer_Name'] == bond_name]
    
    if not bond_data.empty:
        st.subheader(f'Information for bond: {bond_name}')
        
        # Display bond information
        st.write(bond_data)
        
        # Predict risk
        features = model_data['features']
        bond_features = bond_data[features]
        predicted_risk = model_data['model'].predict(bond_features)
        predicted_risk_category = model_data['label_encoder'].inverse_transform(predicted_risk)
        
        st.subheader('Predicted Risk:')
        st.write(predicted_risk_category[0])
        
        # Visualizations
        st.subheader('Bond Characteristics:')
        
        # Bar plot for numeric features
        numeric_features = bond_features.select_dtypes(include=['float64', 'int64']).columns
        fig, ax = plt.subplots(figsize=(10, 6))
        bond_features[numeric_features].iloc[0].plot(kind='bar')
        plt.title('Numeric Features of the Bond')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Pie chart for Coupon_Type
        fig, ax = plt.subplots()
        bond_data['Coupon_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Coupon Type')
        st.pyplot(fig)
        
        # Additional information
        st.subheader('Additional Information:')
        st.write(f"Issuer Sector: {bond_data['Issuer_Sector'].iloc[0]}")
        st.write(f"Issuer Location: {bond_data['Issuer_Location'].iloc[0]}")
        st.write(f"Credit Rating: {bond_data['Credit_Rating'].iloc[0]}")
        
    else:
        st.error(f"Bond '{bond_name}' not found in the dataset.")
else:
    st.info('Please enter a bond name to get information and risk prediction.')

# Add a section to show overall dataset statistics
st.sidebar.title('Dataset Overview')
if st.sidebar.checkbox('Show Dataset Statistics'):
    st.sidebar.subheader('Dataset Shape:')
    st.sidebar.write(data.shape)
    
    st.sidebar.subheader('Data Types:')
    st.sidebar.write(data.dtypes)
    
    st.sidebar.subheader('Summary Statistics:')
    st.sidebar.write(data.describe())

# Add a section to show feature importance
if st.sidebar.checkbox('Show Feature Importance'):
    st.sidebar.subheader('Feature Importance:')
    feature_importance = pd.DataFrame({
        'feature': model_data['features'],
        'importance': model_data['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    st.sidebar.pyplot(fig)