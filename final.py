#!/usr/bin/env python
# coding: utf-8

# In[3]:





# In[4]:


#importing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# In[5]:


#reading dataset
data=pd.read_csv("creditcard.csv")


# In[6]:


#data.tail()


# In[7]:


#finding for fraud==1 && normal==0 ,then differentating it up.
fraud=data.loc[data['Class']==1]
normal=data.loc[data['Class']==0]


# In[8]:


#fraud


# In[11]:


#sns.relplot(x='Amount',y='Time',hue='Class', data=data)


# In[12]:


#data1= data.sample(frac=0.1, random_state = 1)
#print(data1.shape)
#print(data1.describe())


# In[13]:


# Plot histograms of each parameter 
#data.hist(figsize = (20,20))
#plt.show()


# In[14]:


#Plotting the dataset
#plt.plot(data)


# # Using Logistic Regressing

# In[17]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split 


# In[18]:


X=data.iloc[:,:-1]
Y=data['Class']


# In[19]:


X_train,X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.33)


# In[20]:


clf= linear_model.LogisticRegression()


# In[21]:


clf.fit(X_train, Y_train)


# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


# In[22]:


#making model
#y_pred=np.array(clf.predict(X_test))
#y=np.array(Y_test)


# In[23]:


#from sklearn.metrics import confusion_matrix , classification_report ,accuracy_score


# In[24]:


#print(confusion_matrix(Y_test,y_pred))


# In[25]:


#print(accuracy_score(Y_test,y_pred))


# In[26]:


#printing report 
#print(classification_report(Y_test,y_pred))


# In[ ]:





# In[ ]:
'''




