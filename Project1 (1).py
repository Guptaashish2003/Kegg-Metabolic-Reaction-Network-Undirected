#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df=pd.read_csv("D:\\downloads\\Reaction Network.csv",header=None)


# In[3]:


df.head()


# # Loading Header

# In[4]:


df2=pd.read_excel("D:\\downloads\\Reaction Network.xlsx",header=None)
df2.head()


# In[5]:


temp=[]
for i in range(len(df2[0])):
    temp.append(df2[0][i])
print(temp)
df.columns=temp


# In[6]:


df.head()


# In[7]:


df.replace('?',np.nan,inplace=True)


# In[8]:


df.head()


# In[9]:


df.isnull().sum()


# In[10]:


from sklearn.impute import SimpleImputer
obj=SimpleImputer(missing_values=np.nan,strategy='mean')
col=df.iloc[:,4].values.reshape(-1,1)
obj.fit(col)
df.iloc[:, 4]=obj.transform(col)


# In[11]:


df


# In[12]:


unq=df.nunique()
print(unq)


# In[13]:


y=df['Density real'].values
x=df.drop(['Density real','Pathway text'],axis=1).values
print(x)
print(y)


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[15]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train[:,1:]=sc.fit_transform(x_train[:,1:])
x_test[:,1:]=sc.transform(x_test[:,1:])


# In[16]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

reg1 = LinearRegression()
reg2 = SVR(kernel='rbf')
reg3 = DecisionTreeRegressor(max_depth=5)
reg4 = KNeighborsRegressor(n_neighbors=5)

models = [reg1, reg2, reg3, reg4]
model_names = ['linear_regression', 'svr', 'decision_tree', 'knn']

mse_scores = {}
r2_scores = {}
execution_times = {}

for model, model_name in zip(models, model_names):
    start_time = time.time()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    end_time = time.time()
    
    mse_scores[model_name] = mean_squared_error(y_test, y_pred)
    r2_scores[model_name] = r2_score(y_test, y_pred)
    execution_times[model_name] = end_time - start_time

for model_name, mse_score in mse_scores.items():
    print(f"{model_name} MSE: {mse_score} ")

print()

for model_name, r2_score in r2_scores.items():
    print(f"{model_name} R2: {r2_score}")


# In[17]:


import matplotlib.pyplot as plt

# Assuming you have the predicted values in 'y_pred' and actual values in 'y_actual'

# Scatter Plot
plt.scatter(y_pred, y_test)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Regression: Predicted vs Actual')
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Regression: Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')  # Adding a horizontal line at y=0
plt.show()


# In[ ]:




