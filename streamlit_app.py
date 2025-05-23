# -*- coding: utf-8 -*-
"""
Created on friday may 3 22:08:44 2025

@author: surya
"""

import pandas as pd
import joblib
import streamlit as st
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import random  # Add this import for random selection

# Load the preprocessed training data
train_data = pd.read_csv("updated_X.csv")

# Instantiate OneHotEncoder
one_hot = OneHotEncoder()
categorical_values = ['OverallQual', 'GarageCars', 'TotRmsAbvGrd', 'Neighborhood', 'FullBath', 'GarageType']
transformer = ColumnTransformer([('one_hot', one_hot, categorical_values)],
                                  remainder='passthrough')
transformer.fit(train_data)

# Function to transform input data
def transform_data(transformer, data):
    transform_X = transformer.transform(data).toarray()
    transformed_df = pd.DataFrame(transform_X)
    return transformed_df

# Load the trained model
loaded_model = joblib.load("best_model.sav")

# Function to predict house prices
def house_price_prediction(input_data):
    transformed_data = transform_data(transformer, input_data)
    predictions = loaded_model.predict(transformed_data)
    return predictions

# Define meaningful labels for categorical options
options_names = {
    'OverallQual': {
        10: 'Very Excellent',
        9: 'Excellent',
        8: 'Very Good',
        7: 'Good',
        6: 'Above Average',
        5: 'Average',
        4: 'Below Average',
        3: 'Fair',
        2: 'Poor',
        1: 'Very Poor'
    },
    'GarageCars': {
        0: '0 cars',
        1: '1 car',
        2: '2 cars',
        3: '3 cars',
        4: '4 cars'
    },
    'TotRmsAbvGrd': {
        2: '2 rooms',
        3: '3 rooms',
        4: '4 rooms',
        5: '5 rooms',
        6: '6 rooms',
        7: '7 rooms',
        8: '8 rooms',
        9: '9 rooms',
        10: '10 rooms',
        11: '11 rooms',
        12: '12 rooms',
        13: '13 rooms',
        14: '14 rooms'
    },
    'Neighborhood': {
        'NAmes': 'T nagar',
        'CollgCr': 'Anna Nagar',
        'OldTown': 'Chrompet',
        'Edwards': 'Pallavaram',
        'Somerst': 'Saidapet',
        'Gilbert': 'Velachery',
        'NridgHt': 'Nungambakkam',
        'Sawyer': 'Tidel Park',
        'NWAmes': 'Mount Road',
        'SawyerW': 'ECR Road',
        'BrkSide': 'Meenambakkam',
        'Crawfor': 'Chinthadripet',
        'Mitchel': 'Mylapore',
        'NoRidge': 'Guindy',
        'Timber': 'Thiruvanmiyur',
        'IDOTRR': 'Egmore',
        'ClearCr': 'Kelambakkam',
        'StoneBr': 'Thiruvottiyur',
        'SWISU': 'Tambaram',
        'MeadowV': 'Perungudi',
        'Blmngtn': 'Thuraipakkam',
        'BrDale': 'Adyar',
        'Veenker': 'Pallikaranai',
        'NPkVill': 'Vandalur',
        'Blueste': 'Selaiyur',
    },
    'FullBath': {
        0: '0 bathrooms',
        1: '1 bathroom',
        2: '2 bathrooms',
        3: '3 bathrooms'
    },
    'GarageType': {
        '2Types': 'More than one type',
        'Attchd': 'Attached',
        'Basment': 'Basement',
        'BuiltIn': 'Built-In',
        'CarPort': 'Car Port',
        'Detchd': 'Detached',
        'missing': 'No Garage'
    }
}

# Streamlit interface
def main():
    st.markdown(
        """
        <style>
        .stApp {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h1>House Price Prediction</h1>", unsafe_allow_html=True)
    st.write("**by Surya**  \n[GitHub](https://github.com/Suryaseenivasan2005) | [LinkedIn](https://www.linkedin.com/in/surya-seenivasan-b85249354/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BrPcksoz6Q%2BCF%2BMxzxhgfzQ%3D%3D)")
    st.markdown("##### **“The best way to predict the future is to study the past.” — Robert Kiyosaki.**")


    st.sidebar.header("House Details")

    features = {
        'OverallQual': 'Overall Quality',
        'GrLivArea': 'Above Ground Living Area (sqft)',
        'TotalBsmtSF': 'Total Basement Area (sqft)',
        '2ndFlrSF': 'Second Floor Area (sqft)',
        'BsmtFinSF1': 'Storage Area (sqft)',
        '1stFlrSF': 'First Floor Area (sqft)',
        'GarageCars': 'Number of Garage Cars',
        'GarageArea': 'Garage Area (sqft)',
        'LotArea': 'Lot Area (sqft)',
        'TotRmsAbvGrd': 'Total Rooms Above Ground',
        'Age': 'Age of House',
        'Neighborhood': 'Site',
        'YearRemodAdd': 'Year of Remodeling',
        'MasVnrArea': 'Hall (sqft)',
        'BsmtUnfSF': 'Garden(sqft)',
        'FullBath': 'Number of Bathrooms',
        'LotFrontage': 'Lot Frontage (ft)',
        'WoodDeckSF': 'Wood Deck Area (sqft)',
        'GarageYrBlt': 'Year Garage Built',
        'GarageType': 'Garage Type'
    }

    input_data = {}
    for feature, label in features.items():
        if feature == 'OverallQual':
            sorted_options = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
            input_data[feature] = st.sidebar.selectbox(f"{label}:", sorted_options, format_func=lambda x: options_names[feature][x])
        elif feature in ['GarageCars', 'TotRmsAbvGrd', 'FullBath']:
            input_data[feature] = st.sidebar.selectbox(f"{label}:", sorted(train_data[feature].unique()), format_func=lambda x: options_names[feature][x])
        elif feature == 'Neighborhood':
            input_data[feature] = st.sidebar.selectbox(f"{label}:", sorted(train_data[feature].unique()), format_func=lambda x: options_names[feature][x])
        elif feature == 'GarageType':
            input_data[feature] = st.sidebar.selectbox(f"{label}:", sorted(train_data[feature].unique()), format_func=lambda x: options_names[feature].get(x, x))
        else:
            input_data[feature] = st.sidebar.number_input(f"{label}:", min_value=0.0, value=0.0)

    input_df = pd.DataFrame([input_data])

    if st.sidebar.button("Calculate Estimated Price"):
        predicted_price = house_price_prediction(input_df)
        exchange_rate = 82  # Example exchange rate: 1 USD = 82 INR
        predicted_price_inr = predicted_price[0] * exchange_rate
        st.write("### Estimated House Price: ₹", round(predicted_price_inr, 2))
        
        # List of images to display
        images = ["img_1.jpg", "img_2.jpg", "img_3.jpg","img_4.jpg","img_5.jpg","img_6.jpg","img_7.jpg"]  # Add your image file names here
        selected_image = random.choice(images)  # Randomly select an image
        image = Image.open(selected_image)
        st.image(image, caption="House Image")

if __name__ == "__main__":
    main()
