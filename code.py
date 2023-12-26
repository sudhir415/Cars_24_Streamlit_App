import pandas as pd 
import numpy as np 
import streamlit as st
import datetime
import pickle 


cars_df = pd.read_csv("./cars24-car-price.csv") 


st.write(""" 
         # Cars24 Used Car Price Prediction
         """) 

st.dataframe(cars_df.head()) 

encode_dict = {
            "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric": 5}, 
            "seller_type": {"Dealer": 1,"Individual": 2,"Trustmark Dealer": 3}, 
            "transmission_type": {"Manual": 1, "Automatic": 2}
} 

def model_pred(year, seller_type, km_driven, fuel_type, transmission_type, mileage, engine, max_power, seats) :

    ## Loading the Model 
    with open("car_pred", "rb") as file :
        reg_model = pickle.load(file) 

    # Pass the inferences like what are the inferences pass here -- 
    input_features = [[year, seller_type, km_driven, fuel_type, transmission_type, mileage, engine, max_power, seats]] 
    return reg_model.predict(input_features) 

col1, col2 = st.columns(2) 

fuel_type = col1.selectbox("Select fuel type", 
                           ["Diesel", "Petrol", "CNG", "LPG", "Electric"])

transmission_type = col1.selectbox("Select transmission type",
                                    ['Manual', "Automatic"]) 

seats = col1.selectbox("No. of Seats",
                        [4,5,6,7])
seller_type = col1.selectbox("Choose the seller type", 
                             ['Dealer', 'Individual', 'Trustmark Dealer']) 

engine = col2.slider("Set the engine power",
                        500, 5000, step=100) 
year = col2.slider("Set the year", 2005.0, 2020.0, step= 1.0) 

km_driven = col2.slider("Set the km_driven", 10000, 100000, step= 1000)

mileage = col2.slider("Set the mileage", 5.0, 30.0, step= 1.0)

max_power = col2.slider("Power in HP select", 50.0, 200.0, step= 10.0)


if(st.button("Predict Price")):
    fuel_type = encode_dict['fuel_type'][fuel_type]
    transmission_type = encode_dict['transmission_type'][transmission_type]
    seller_type = encode_dict["seller_type"][seller_type]
    price = model_pred(year, seller_type, km_driven, fuel_type, transmission_type, mileage, engine, max_power,seats)
    st.text("Predicted price of the car: "+ str(price)) 



