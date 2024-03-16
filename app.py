import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model_file = 'model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # List of columns in the desired order
    ordered_columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                       'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 
                       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
                       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
                       'Temp9am', 'Temp3pm', 'RainToday']

    # Ensure input data has the desired order of columns
    data = data.reindex(columns=ordered_columns)

    # Mapping wind direction labels to numerical values
    wind_dir_mapping = {
        'N': 0, 'NNE': 1, 'NE': 2, 'ENE': 3, 'E': 4, 'ESE': 5, 'SE': 6, 'SSE': 7,
        'S': 8, 'SSW': 9, 'SW': 10, 'WSW': 11, 'W': 12, 'WNW': 13, 'NW': 14, 'NNW': 15
    }

    # Apply mapping to wind direction columns
    data['WindGustDir'] = data['WindGustDir'].map(wind_dir_mapping)
    data['WindDir9am'] = data['WindDir9am'].map(wind_dir_mapping)
    data['WindDir3pm'] = data['WindDir3pm'].map(wind_dir_mapping)

    # Convert 'RainToday' to binary
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
    
    return data

# Function to predict
def predict(data):
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)
    return prediction

# Streamlit UI
st.title('Rainfall Detection')
st.write('Enter the following details to predict rainfall tomorrow:')

location_options = ['Portland', 'Cairns', 'Walpole', 'Dartmoor', 'MountGambier',
       'NorfolkIsland', 'Albany', 'Witchcliffe', 'CoffsHarbour', 'Sydney',
       'Darwin', 'MountGinini', 'NorahHead', 'Ballarat', 'GoldCoast',
       'SydneyAirport', 'Hobart', 'Watsonia', 'Newcastle', 'Wollongong',
       'Brisbane', 'Williamtown', 'Launceston', 'Adelaide', 'MelbourneAirport',
       'Perth', 'Sale', 'Melbourne', 'Canberra', 'Albury', 'Penrith',
       'Nuriootpa', 'BadgerysCreek', 'Tuggeranong', 'PerthAirport', 'Bendigo',
       'Richmond', 'WaggaWagga', 'Townsville', 'PearceRAAF', 'SalmonGums',
       'Moree', 'Cobar', 'Mildura', 'Katherine', 'AliceSprings', 'Nhil',
       'Woomera', 'Uluru']

# Mapping locations to numerical values
location_mapping = {location: idx+1 for idx, location in enumerate(location_options)}

location = st.selectbox('Location:', location_options)

min_temp = st.number_input('Min Temperature (°C):', value=0.0)
max_temp = st.number_input('Max Temperature (°C):', value=0.0)
rainfall = st.number_input('Rainfall (mm):', value=0.0)
evaporation = st.number_input('Evaporation (mm):', value=0.0)
sunshine = st.number_input('Sunshine (hours):', value=0.0)

wind_gust_dir = st.selectbox('Wind Gust Direction:', ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                                                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])

wind_gust_speed = st.number_input('Wind Gust Speed (km/h):', value=0.0)

wind_dir_9am = st.selectbox('Wind Direction at 9am:', ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                                                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])

wind_dir_3pm = st.selectbox('Wind Direction at 3pm:', ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                                                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])

wind_speed_9am = st.number_input('Wind Speed at 9am (km/h):', value=0.0)
wind_speed_3pm = st.number_input('Wind Speed at 3pm (km/h):', value=0.0)

humidity_9am = st.slider('Humidity at 9am:', 0, 100, 50)
humidity_3pm = st.slider('Humidity at 3pm:', 0, 100, 50)
pressure_9am = st.number_input('Pressure at 9am (hpa):', value=0.0)
pressure_3pm = st.number_input('Pressure at 3pm (hpa):', value=0.0)
cloud_9am = st.slider('Cloud at 9am:', 0, 8, 4)
cloud_3pm = st.slider('Cloud at 3pm:', 0, 8, 4)
temp_9am = st.number_input('Temperature at 9am (°C):', value=0.0)
temp_3pm = st.number_input('Temperature at 3pm (°C):', value=0.0)
rain_today = st.selectbox('Rain Today:', ['Yes', 'No'])

input_data = pd.DataFrame({
    'Location': [location_mapping[location]],
    'MinTemp': [min_temp],
    'MaxTemp': [max_temp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'WindGustDir': [wind_gust_dir],
    'WindGustSpeed': [wind_gust_speed],
    'WindDir9am': [wind_dir_9am],
    'WindDir3pm': [wind_dir_3pm],
    'WindSpeed9am': [wind_speed_9am],
    'WindSpeed3pm': [wind_speed_3pm],
    'Humidity9am': [humidity_9am],
    'Humidity3pm': [humidity_3pm],
    'Pressure9am': [pressure_9am],
    'Pressure3pm': [pressure_3pm],
    'Cloud9am': [cloud_9am],
    'Cloud3pm': [cloud_3pm],
    'Temp9am': [temp_9am],
    'Temp3pm': [temp_3pm],
    'RainToday': [rain_today]
})

if st.button('Predict'):
    prediction = predict(input_data)
    if prediction[0] == 1:
        st.write('It will rain tomorrow!')
    else:
        st.write("It won't rain tomorrow!")