import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import streamlit as st

df = pd.read_csv('Student_Performance.csv')
df.head()

sns.jointplot(x = 'Previous Scores',y = 'Performance Index',data = df)
sns.pairplot(df,diag_kws={'alpha':0.4})

sns.lmplot(x='Previous Scores',y = 'Performance Index',data=df,scatter_kws={'alpha':0.5})
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'True':1,'False':0})

df.head()
X = df[['Previous Scores','Hours Studied']]
y = df['Performance Index']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)

predicted = lr.predict(X_test)
plt.figure(figsize=(12,6))
plt.scatter(predicted,y_test)

print("Mean Squared Error ",mean_squared_error(y_test,predicted))
print("Mean Absolute Error ",mean_absolute_error(y_test,predicted))
print("Root Mean Square Error ",math.sqrt(mean_squared_error(y_test,predicted)))

def predict_performance(previous_scores, hours_studied):
    data = pd.DataFrame({'Previous Scores': [previous_scores], 'Hours Studied': [hours_studied]})
    prediction = lr.predict(data)[0]
    return prediction
    
st.title('Student Performance Prediction App')
previous_scores = st.number_input('Enter Previous Scores')
hours_studied = st.number_input('Enter Hours Studied')

if st.button('Predict Performance'):
    prediction = predict_performance(previous_scores, hours_studied)
    st.write('Predicted Performance Index:', prediction)



