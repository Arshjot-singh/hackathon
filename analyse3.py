import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv(r'D:\Desktop\BlackRock\synthetic_green_bond_data_with_real_issuers (2).csv')

data = load_data()

# Set up the Streamlit app
st.title('Green Bond Information')

# User input
bond_name = st.text_input('Enter the bond name:')

if bond_name:
    # Find the bond in the dataset
    bond_data = data[data['Issuer_Name'] == bond_name]
    
    if not bond_data.empty:
        st.subheader(f'Information for bond: {bond_name}')
        
        # Display bond information
        st.write(bond_data.drop(columns=['Risk_Involved'], errors='ignore'))
        
        # Visualizations
        st.subheader('Bond Characteristics:')
        
        # Bar plot for numeric features
        numeric_features = bond_data.select_dtypes(include=['float64', 'int64']).columns
        numeric_features = numeric_features[numeric_features != 'Risk_Involved']
        fig, ax = plt.subplots(figsize=(10, 6))
        bond_data[numeric_features].iloc[0].plot(kind='bar')
        plt.title('Numeric Features of the Bond')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional information
        st.subheader('Additional Information:')
        for col in bond_data.columns:
            if col not in numeric_features and col not in ['Issuer_Name', 'Risk_Involved']:
                st.write(f"{col}: {bond_data[col].iloc[0]}")
        
    else:
        st.error(f"Bond '{bond_name}' not found in the dataset.")
else:
    st.info('Please enter a bond name to get information.')

# Add a section to show overall dataset statistics
st.sidebar.title('Dataset Overview')
if st.sidebar.checkbox('Show Dataset Statistics'):
    st.sidebar.subheader('Dataset Shape:')
    st.sidebar.write(data.shape)
    
    st.sidebar.subheader('Data Types:')
    st.sidebar.write(data.dtypes)
    
    st.sidebar.subheader('Summary Statistics:')
    st.sidebar.write(data.describe())

# Add a section to show feature distributions
if st.sidebar.checkbox('Show Feature Distributions'):
    st.sidebar.subheader('Feature Distributions:')
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_features = numeric_features[numeric_features != 'Risk_Involved']
    feature = st.sidebar.selectbox('Select a feature:', options=numeric_features)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    st.sidebar.pyplot(fig)