import streamlit as st
import numpy as np
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# LOAD DATASET
iris = load_iris()

x = iris.data
y = iris.target


# SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# CREATE SVM MODEL
model = SVC(kernel='linear')


# TRAIN MODEL
model.fit(x_train, y_train)


# SAVE MODEL
pickle.dump(model, open("svm_model.pkl", "wb"))


# LOAD MODEL
loaded_model = pickle.load(open("svm_model.pkl", "rb"))


# STREAMLIT TITLE
st.title("SVM Classifier Application")

st.write("Enter Iris Flower Measurements")


# USER INPUTS
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")


# PREDICT BUTTON
if st.button("Predict"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = loaded_model.predict(input_data)

    if prediction[0] == 0:
        st.success("Predicted Flower : Setosa")

    elif prediction[0] == 1:
        st.success("Predicted Flower : Versicolor")

    else:
        st.success("Predicted Flower : Virginica")