# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st

# import data
df = pd.read_excel('cars.xls', engine='openpyxl')
x=df.drop('Price', axis=1)
y=df[['Price']]

# train test split veriyi ikiye ayirma
x_train,x_test, y_train, y_test=train_test_split(x,y, random_state=42, test_size=0.20)

# preprocessing - ön işleme

preprocessor=ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),['Mileage','Cylinder','Liter','Doors']),
        ('cat',OneHotEncoder(),['Make','Model','Trim','Type'])
    ]
    
)

# Model Tanimlama
model=LinearRegression()

# Pipeline
pipeline=Pipeline(steps=[('preprocessor',preprocessor),('regressor',model)])

# training - Modeli Egitme
pipeline.fit(x_train,y_train)

# Predict unseen data - Tahmin et
pred=pipeline.predict(x_test)

# calculate scores, hata ve basari oranini hesapla
rmse=mean_squared_error(y_test,pred)**0.5
r2=r2_score(y_test,pred)

# prediction function
def price_pred(make, model, trim, mileage, type_, cylinder, liter, doors, cruise, sound, leather):
    input_data = pd.DataFrame({
        'Make': [make],
        'Model': [model],
        'Trim': [trim],
        'Mileage': [mileage],
        'Type': [type_],
        'Cylinder': [cylinder],
        'Liter': [liter],
        'Doors': [doors],
        'Cruise': [cruise],
        'Sound': [sound],
        'Leather': [leather]
    })
    prediction = pipeline.predict(input_data)[0]
    return prediction

st.title('MLOps Car Price Prediction App :red_car:')
st.write('Enter Car Details to Prdict the Price')

make=st.selectbox("Make",df['Make'].unique())
carmodel=st.selectbox("Model",df[df['Make']==make]['Model'].unique())
trim = st.selectbox("Trim", df[df["Model"] == carmodel]["Trim"].unique())
mileage = st.number_input('Mileage', min_value=200, max_value=60000, step=100)
car_type = st.selectbox('Type', df['Type'].unique())
cylinders = st.selectbox('Cylinder', df['Cylinder'].unique())
liter = st.number_input('Liter', min_value=1, max_value=6, step=1)
doors = st.selectbox('Doors', df['Doors'].unique())
cruise = st.radio('Cruise', [0, 1])
sound = st.radio('Sound', [0, 1])
leather = st.radio('Leather', [0, 1])

if st.button('Predict'):
    predicted_price = price_pred(make, carmodel, trim, mileage, car_type, cylinders, liter, doors, cruise, sound, leather)
    price=float(predicted_price)
    st.success(f'The predicted price is: ${price:.2f}')





