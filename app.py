import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.header('🚗 Car Price Prediction ML Model')

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('Number of Kilometers Driven', 11, 200000)
fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
owner = st.selectbox('Owner Type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (km/l)', 10, 40)
engine = st.slider('Engine Capacity (CC)', 700, 5000)
max_power = st.slider('Maximum Power (HP)', 0, 200)
seats = st.slider('Number of Seats', 2, 10)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    mappings = {
        'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 
                  'Fourth & Above Owner': 4, 'Test Drive Car': 5},
        'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
        'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
        'transmission': {'Manual': 1, 'Automatic': 2},
        'name': {brand: i+1 for i, brand in enumerate(cars_data['name'].unique())}
    }

    for col, mapping in mappings.items():
        input_data[col].replace(mapping, inplace=True)

    car_price = model.predict(input_data)

    st.success(f'💰 Predicted Car Price: ₹ {car_price[0]:,.2f}')