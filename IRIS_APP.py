#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
import numpy as np


# In[5]:


model = pickle.load(open("iris_model.pkl",'rb'))


# In[6]:


# Title of the app
st.title("Iris Flower Prediction App")


# In[7]:


# Input features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0)


# In[8]:


# When the user clicks the button, make a prediction
if st.button("Predict"):
    # Prepare the input for the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    # Display the prediction
    st.write(f"The predicted class is: {prediction[0]}")


# In[ ]:




