#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pip install sklearn


# In[3]:


import sklearn


# In[4]:


from sklearn import datasets
boston_df=datasets.load_boston()


# In[5]:


boston_df.keys()


# In[6]:


print(boston_df.DESCR)


# In[7]:


print(boston_df.data)


# In[8]:


print(boston_df.target)


# In[9]:


print(boston_df.feature_names)


# In[10]:


dataset=pd.DataFrame(boston_df.data,columns=boston_df.feature_names)


# In[11]:


dataset.head()


# In[12]:


dataset['Price']=boston_df.target


# In[13]:


dataset.head()


# In[14]:


dataset.info()


# In[15]:


dataset.describe()


# In[16]:


dataset.isnull().sum()


# In[17]:


dataset.corr()


# In[18]:


import seaborn as sns
sns.regplot(x="RM",y="Price",data=dataset)


# In[19]:


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[20]:


x


# In[21]:


y


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[23]:


x_train


# In[24]:


x_test


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[26]:


x_train=scaler.fit_transform(x_train)

x_test=scaler.transform(x_test)
# In[27]:


x_train


# In[28]:


x_test


# In[29]:


x_test=scaler.transform(x_test)


# In[30]:


x_test


# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


regression=LinearRegression()
regression.fit(x_train,y_train)


# In[33]:


print(regression.coef_)


# In[34]:


regression.get_params()


# In[35]:


reg_pred=regression.predict(x_test)


# In[36]:


reg_pred


# In[37]:


plt.scatter(y_test,reg_pred)


# In[38]:


residual=y_test-reg_pred


# In[39]:


residual


# In[41]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))


# In[42]:


print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[43]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# In[44]:


1-(1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# In[49]:


boston_df.data[0].reshape(1,-1)


# In[50]:


scaler.transform(boston_df.data[0].reshape(1,-1))


# In[52]:


regression.predict(scaler.transform(boston_df.data[0].reshape(1,-1)))


# In[53]:


import pickle


# In[54]:


pickle.dump(regression,open('regmodel.pkl','wb'))


# In[55]:


pickled_model=pickle.load(open('regmodel.pkl','rb'))


# In[57]:


pickled_model.predict(scaler.transform(boston_df.data[0].reshape(1,-1)))


# In[ ]:




