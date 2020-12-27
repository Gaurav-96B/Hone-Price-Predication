#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('homeprices.csv')
df


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[4]:


new_df = df.drop('price',axis='columns')
new_df


# In[5]:


price = df.price
price


# In[6]:


reg = linear_model.LinearRegression()
reg.fit(new_df,price)


# In[7]:


reg.predict([[3300]])


# In[8]:


reg.coef_


# In[9]:


reg.intercept_


# In[10]:


3300*135.78767123 + 180616.43835616432


# In[11]:


reg.predict([[5000]])


# In[12]:


area_df = pd.read_csv("areas.csv")
area_df


# In[13]:


p = reg.predict(area_df)
p


# In[14]:


area_df['prices']=p
area_df

