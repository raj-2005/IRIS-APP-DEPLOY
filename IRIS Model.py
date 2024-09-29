#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


# In[2]:


# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[3]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


# Train the model
model = SVC(kernel='linear')
model.fit(X_train, y_train)


# In[5]:


# Save the model
pickle.dump(model, open('iris_model.pkl', 'wb'))


# In[ ]:




