#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Social_Network_Ads.csv")
data.head(7)


# In[3]:


data.describe()


# In[4]:


sns.pairplot(data)


# In[5]:


Sex = pd.get_dummies(data['Gender'], drop_first = True)
Sex


# In[6]:


data['Sex'] = Sex
data = data.drop('Gender' , axis =1)


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


ss = StandardScaler()
ss.fit(data.drop('Purchased', axis =1))


# In[9]:


scaled_featured = ss.transform(data.drop('Purchased', axis=1))
scaled_featured


# In[10]:


scale = pd.DataFrame(scaled_featured, columns = data.columns[:-1])
scale['Sex'] = scale['Purchased']
scale = scale.drop('Purchased' ,axis=1)
scale


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scale,data['Purchased'],test_size = 0.3 , random_state = 50)


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# In[15]:


#KNN
knn = KNeighborsClassifier( n_neighbors = 5)
knn.fit(x_train , y_train)


# In[16]:


pred = knn.predict(x_test)


# In[17]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix (y_test, pred))


# In[18]:


print(classification_report(y_test , pred))


# In[19]:


#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train, y_train)


# In[20]:


pred = dtc.predict(x_test)


# In[21]:


print(classification_report(y_test, pred))


# In[22]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=225,random_state=1)
rfc.fit(x_train, y_train)
pred = rfc.predict(x_test)
print(classification_report(y_test, pred))


# In[23]:


#Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train, y_train)


# In[24]:


pred= nb.predict(x_test)


# In[25]:


print(classification_report(y_test, pred))


# In[26]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[27]:


pred = lr.predict(x_test)


# In[28]:


print(classification_report(y_test, pred))


# In[29]:


#SVM Classification
from sklearn.svm import SVC
svc=SVC(random_state=1)
svc.fit(x_train,y_train)


# In[30]:


pred = svc.predict(x_test)


# In[31]:


print(classification_report(y_test, pred))


# In[33]:


#Gradient Boosting
gfc=GradientBoostingClassifier(n_estimators =1000, max_leaf_nodes= 4, max_depth=None,random_state= 2,min_samples_split= 5)
gfc.fit(x_train,y_train)


# In[34]:


pred = gfc.predict(x_test)


# In[35]:


print(classification_report(y_test, pred))


# In[36]:


#AdaBoost
abc=AdaBoostClassifier(n_estimators=100, random_state=0)
abc.fit(x_train,y_train)


# In[37]:


pred=abc.predict(x_test)


# In[38]:


print(classification_report(y_test, pred))


# In[39]:


ysa=MLPClassifier(alpha=1,max_iter=1000)
ysa.fit(x_train,y_train)


# In[40]:


pred = ysa.predict(x_test)


# In[41]:


print(classification_report(y_test, pred))


# In[44]:


get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)


# In[45]:


pred = model.predict(x_test)


# In[46]:


print(classification_report(y_test, pred))


# In[ ]:




