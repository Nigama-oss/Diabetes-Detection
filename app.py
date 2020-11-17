import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write ("""
# Diabetes Detection Model
This is a machine learning model to detect if someone has diabetes.
""")

image = Image.open('E:/Programming/Python Projects/Diabetes Detection/images/dd.png')
st.image(image, caption = 'ML', use_column_width = True)

df = pd.read_csv('diabetes.csv')

st.subheader('Data Information')
st.dataframe(df)
st.write(df.describe())

chart = st.bar_chart(df)

x = df.iloc[:, 0:8].values
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    user_data = {
    'Pregnancies' : pregnancies,
    'Glucose' : glucose,
    'BloodPressure' : blood_pressure,
    'SkinThickness' : skin_thickness,
    'Insulin' : insulin,
    'BMI' : BMI,
    'DiabetesPedigreeFunction' : DPF,
    'Age' : age
    }

    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

st.subheader('User Input: ')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train, y_train)

st.subheader('Model Accuracy Score')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100) + '%')

prediction = RandomForestClassifier.predict(user_input)

st.subheader('Classification: ')
if prediction == 1:
    st.write(prediction)
    st.write('You have a risk of getting diabetes.')
    st.subheader('Here are a few tips:')
    st.markdown('- Have a diet high in fresh, nutritious foods, including whole grains, fruits, vegetables, lean proteins, low-fat dairy, and healthy fat sources, such as nuts.')
    st.markdown('- Avoid high-sugar foods that provide empty calories, or calories that do not have other nutritional benefits, such as sweetened sodas, fried foods, and high-sugar desserts.')
    st.markdown('- Refrain from drinking excessive amounts of alcohol or keeping intake to less than one drink a day for women or two drinks a day for men.')
    st.markdown('- Engage in at least 30 minutes exercise a day on at least 5 days of the week, such as of walking, aerobics, riding a bike, or swimming.')
    st.markdown('- Recognize signs of low blood sugar when exercising, including dizziness, confusion, weakness, and profuse sweating.')
else:
    st.write('You are not prone to get diabetes :)')
