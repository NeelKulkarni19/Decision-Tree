#!/usr/bin/env python
# coding: utf-8

# In[70]:


# Question 1 company dataset


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


company= pd.read_csv('F:/Dataset/Company_Data.csv')


# In[4]:


company


# In[5]:


company.info()


# In[6]:


company.corr()


# In[8]:


import seaborn as sns


# In[9]:


sns.jointplot(company['Sales'],company['Income'])


# In[11]:


company.loc[company['Sales']<= 10.00,'Sales1']='Not High'


# In[12]:


company.loc[company['Sales']>=10.01,'Sales1']='High'


# In[13]:


company


# In[14]:


from sklearn import preprocessing


# In[15]:


lable_encoder= preprocessing.LabelEncoder()


# In[21]:


company['ShelveLoc']= lable_encoder.fit_transform(company['ShelveLoc'])


# In[22]:


company['Urban']= lable_encoder.fit_transform(company['Urban'])


# In[23]:


company['US']= lable_encoder.fit_transform(company['US'])


# In[24]:


company['Sales1']= lable_encoder.fit_transform(company['Sales1'])


# In[25]:


company


# In[29]:


x= company.iloc[:,1:11]


# In[30]:


y= company['Sales1']


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.tree import  DecisionTreeClassifier


# In[35]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=50)


# In[36]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)


# In[37]:


model.fit(x_train,y_train)


# In[38]:


model.get_n_leaves()


# In[41]:


preds = model.predict(x_test) 


# In[42]:


pd.Series(preds).value_counts()


# In[43]:


preds


# In[44]:


pd.crosstab(y_test,preds)


# In[45]:


np.mean(preds==y_test)


# In[46]:


print(classification_report(preds,y_test))


# In[47]:


modelgini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[48]:


modelgini.fit(x_train, y_train)


# In[50]:


modelgini.get_n_leaves()


# In[52]:


preds = modelgini.predict(x_test)


# In[53]:


pd.Series(preds).value_counts()


# In[54]:


preds


# In[55]:


np.mean(preds==y_test)


# In[56]:


print(classification_report(preds,y_test))


# In[57]:


from sklearn.tree import DecisionTreeRegressor


# In[60]:


model1 = DecisionTreeRegressor()


# In[61]:


model1.fit(x_train, y_train)


# 

# In[62]:


preds = model1.predict(x_test) 


# In[63]:


np.mean(preds==y_test)


# In[64]:


Plot = plt.figure(figsize=(25,20))


# In[65]:


Plot = tree.plot_tree(model)


# In[66]:


Fig = plt.figure(figsize=(25,20))


# In[67]:


Fig = tree.plot_tree(modelgini)


# In[68]:


fig = plt.figure(figsize=(25,20))


# In[69]:


fig = tree.plot_tree(model1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[71]:


# Question 2 Fraud Dataset


# In[77]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.tree import  DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# In[78]:


fraud= pd.read_csv('F:/Dataset/Fraud_check.csv')


# In[79]:


fraud


# In[80]:


fraud.info()


# In[81]:


fraud.corr()


# In[82]:


fraud.loc[fraud["Taxable.Income"] <= 30000,"Taxable_Income"]="Good"


# In[83]:


fraud.loc[fraud["Taxable.Income"] > 30001,"Taxable_Income"]="Risky"


# In[84]:


fraud


# In[85]:


lable_encoder=preprocessing.LabelEncoder()


# In[86]:


fraud["Undergrad"]=lable_encoder.fit_transform(fraud["Undergrad"])


# In[87]:


fraud["Marital.Status"]=lable_encoder.fit_transform(fraud["Marital.Status"])


# In[88]:


fraud["Urban"]=lable_encoder.fit_transform(fraud["Urban"])


# In[89]:


fraud["Taxable_Income"]=lable_encoder.fit_transform(fraud["Taxable_Income"])


# In[90]:


fraud


# In[93]:


fraud.drop(["City.Population"],axis=1,inplace=True)


# In[94]:


fraud


# In[96]:


fraud.drop(["Taxable.Income"],axis=1,inplace=True)


# In[98]:


fraud["Taxable_Income"].unique()


# In[99]:


fraud


# In[100]:


x=fraud.iloc[:,0:4]


# In[101]:


x


# In[102]:


y=fraud['Taxable_Income']


# In[103]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[106]:


model = DecisionTreeClassifier(criterion = 'entropy')


# In[107]:


model.fit(x_train,y_train)


# In[109]:


model.get_n_leaves()


# In[110]:


preds = model.predict(x_test)


# In[112]:


pd.Series(preds).value_counts()


# In[113]:


preds


# In[114]:


np.mean(preds==y_test)


# In[115]:


print(classification_report(preds,y_test))


# In[116]:


model_gini = DecisionTreeClassifier(criterion='gini')


# In[117]:


model_gini.fit(x_train, y_train)


# In[118]:


model_gini.get_n_leaves()


# In[122]:


preds = model_gini.predict(x_test) 


# In[123]:


pd.Series(preds).value_counts()


# In[124]:


preds


# In[125]:


np.mean(preds==y_test)


# In[126]:


model_R = DecisionTreeRegressor()


# In[127]:


model_R.fit(x_train, y_train)


# In[128]:


preds = model_R.predict(x_test) 


# In[129]:


np.mean(preds==y_test)


# In[130]:


fig = plt.figure(figsize=(25,20))


# In[131]:


fig = tree.plot_tree(model)


# In[132]:


fig = plt.figure(figsize=(25,20))


# In[133]:


fig = tree.plot_tree(model_gini)


# In[134]:


fig = plt.figure(figsize=(25,20))


# In[135]:


fig = tree.plot_tree(model_R)


# In[ ]:




