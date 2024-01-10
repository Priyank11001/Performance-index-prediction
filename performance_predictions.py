#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[69]:


df = pd.read_csv('Student_Performance.csv')
df.head()


# In[70]:


sns.jointplot(x = 'Previous Scores',y = 'Performance Index',data = df)


# In[71]:


sns.pairplot(df,diag_kws={'alpha':0.4})


# In[72]:


sns.lmplot(x='Previous Scores',y = 'Performance Index',data=df,scatter_kws={'alpha':0.5})
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'True':1,'False':0})


# In[73]:


df.head()
X = df[['Previous Scores','Hours Studied']]
y = df['Performance Index']


# In[74]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[75]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)


# In[86]:


predicted = lr.predict(X_test)
plt.figure(figsize=(12,6))
plt.scatter(predicted,y_test)


# In[93]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
print("Mean Squared Error ",mean_squared_error(y_test,predicted))
print("Mean Absolute Error ",mean_absolute_error(y_test,predicted))
print("Root Mean Square Error ",math.sqrt(mean_squared_error(y_test,predicted)))



# In[ ]:
import streamlit as st

def predict_performance(previous_scores, hours_studied):
    data = pd.DataFrame({'Previous Scores': [previous_scores], 'Hours Studied': [hours_studied]})
    prediction = lr.predict(data)[0]
    return prediction

# UI elements
st.title('Student Performance Prediction App')
previous_scores = st.number_input('Enter Previous Scores')
hours_studied = st.number_input('Enter Hours Studied')

if st.button('Predict Performance'):
    prediction = predict_performance(previous_scores, hours_studied)
    st.write('Predicted Performance Index:', prediction)



