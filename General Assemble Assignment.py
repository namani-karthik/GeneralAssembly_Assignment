#!/usr/bin/env python
# coding: utf-8

# # Part-1

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


# Function to retrieve Column Headers in List
def Reading_Headers():
    f_col_headers=open('field_names.txt','r')
    col_headers=[]
    for i in f_col_headers.readlines():
        if i:
            i=i.strip("\n")
            col_headers.append(i)
    return col_headers


# #### Reading_Headers is a function to retrieve the Column headers from the text file

# In[3]:


import numpy
import matplotlib.pyplot as plot
def bootstrap(index,sample_size):
    sample_list=[]
    for i in range(0,sample_size):
        sample=numpy.random.choice(index,size=1)
        sample_list.append(int(sample))
    return sample_list


# #### BootStrap function to draw the samples. Here logic is, I am redrawing the index number as the sample_size input.
# and returning the sample_list which consists of the index numbers

# In[4]:


import pandas as pd

#Collecting Column headers into a List
col_headers=Reading_Headers()


#Reading breast-cancer data and creating dataframe with respective column headers
df=pd.read_csv('breast-cancer.csv',names=col_headers) #dataframe with actual data for any refernce
d=pd.read_csv('breast-cancer.csv',names=col_headers)

#As diagnosis is the label feature of the data set and as given in categorial type we need to change it to numerical type
d.diagnosis=[1 if each=="M" else 0 for each in df.diagnosis]
y=d['diagnosis']

# droping the diagnosis column from the dataframe
#d.drop(['ID'],axis=1,inplace=True)


smooth_mean=d.groupby(['diagnosis'])['smoothness_mean'].mean()
smooth_median=d.groupby(['diagnosis'])['smoothness_mean'].median()
compact_mean=d.groupby(['diagnosis'])['compactness_mean'].mean()
compact_median=d.groupby(['diagnosis'])['compactness_mean'].median()
print('Smooth Mean={},Smooth Median={},compact Mean={},compact Median={}'.format(smooth_mean,smooth_median,compact_mean,compact_median))


# ###### Program to read the column headers and save the csv file into DataFrame df.
# code to analysis the mean and medians of the respective columns as part of the assignment
# 
# #Comment: Yes Mean and median differ in both cases. Smoothness mean and median as well as Compactness mean and median values are calucalulated based on the mean values given in the dataset using 'groupby on diagnosis column' of dataframe, so that we can get different values for Malignant and Benign cancers
# 
# ##### Observation: Smoothness: Mean is larger than Median for both Malignant and Benign Cancers, which indicates data is right skewed
# 
# #### Observation: Compactness: Mean is slightly larger than Median for both Malignant and Benign Cancers, which indicates data is slightly right skewed

# In[5]:


from scipy.stats import pearsonr
for i in d.columns:
    print(pearsonr(d[i],d['diagnosis'])[0],':Correlation Coefficient for feature',i,' and diagnosis')


# ###### code to determine the correlation coef.

# In[6]:


from sklearn import preprocessing
labels=['fractal_dimension_mean','concave_points_sd_error','perimeter_sd_error','diagnosis']
standardized_data=pd.DataFrame()
for i in labels:
    standardized_data[i]=preprocessing.minmax_scale(d[i])
standardized_data


# ##### Creating a labels of the  feature with highest corr values

# In[7]:


import seaborn as sns
labels=['fractal_dimension_mean','concave_points_sd_error','perimeter_sd_error']
for i in labels:
    ax=sns.scatterplot(d[i],d['diagnosis'])
    ax.set(xlabel=i,ylabel='diagnosis')
    plt.show()


# ###### Visual representation of the features and diagnosis

# Above scatter plot clearly indicating that as the value of feature is increasing there is a clear inclination to diagnosis value to be 1, which is intern representation for Cancer to Malignant.

# In[8]:


from sklearn.model_selection import train_test_split
bootstrap_index=bootstrap(standardized_data.index,3000)
standardized_bootstraped_data=pd.DataFrame(columns=col_headers)
j=0
for i in bootstrap_index:
    standardized_bootstraped_data.loc[j]=(d.iloc[i])
    j+=1

y=pd.DataFrame(standardized_bootstraped_data['diagnosis'])
# droping the diagnosis column from the dataframe
standardized_bootstraped_data.drop(['diagnosis'],axis=1,inplace=True)
standardized_bootstraped_data.drop(['ID'],axis=1,inplace=True)

standardized_bootstraped_Malignant_features_data=pd.DataFrame()
standardized_bootstraped_Malignant_features_data['fractal_dimension_mean']=standardized_bootstraped_data['fractal_dimension_mean']
standardized_bootstraped_Malignant_features_data['concave_points_sd_error']=standardized_bootstraped_data['concave_points_sd_error']
standardized_bootstraped_Malignant_features_data['perimeter_sd_error']=standardized_bootstraped_data['perimeter_sd_error']


# ###### Based on the index values returned by the bootstrap function, standardized_bootstraped_data is constructed. 'diagnosis' and 'ID' columns are dropped. New dataframe standardized_bootstraped_Malignant_features_data is constructed

# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(standardized_bootstraped_Malignant_features_data,y,test_size=0.2,random_state=42)


# ###### spliting of data in Training and Testing

# In[10]:


from sklearn.svm import SVC
clf=SVC(gamma='auto')
clf.fit(X_train,y_train)


# ###### Here I am using Support Vector Machine algorithm as it very effective in classifing the binary classification. With this model my training data accuracy was about 92 % and test data accuracy was also 91.6%

# In[11]:


clf.score(X_train,y_train)


# In[12]:


clf.score(X_test,y_test)


# # Evaluation of the data with second model (Neural Network - Tensorflow)

# In[13]:


import tensorflow as tf
import tensorflow.keras as keras

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu',input_shape=[len(X_train.keys())]),
    keras.layers.Dense(64, activation='relu',input_shape=[len(X_train.keys())]),
    keras.layers.Dense(2, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15)


# #### Using three layers with hidden layer.

# In[14]:


model_predict=model.predict([[0.910653,0.668310,0.755467]])
model_predict


# #### Prediction for this input value is accurate, which is 91.6% chances for malignant Cancer. Actual data also reflect this as Malignant

# # Comparison between two models:
# ## SVM: This algo with work good with binary classification, and become complex with multi classification
# ## Nueral Network: Works fine with multi-classification, consumes lot of hardware resources
# 
# # Performance:
# ## SVM achieved 92% both in training and testing
# ## Neural Netword achieved 91.46 %
# 
# ## with the help of correlation coef values considered three features in both the models.

# # Part - 2

# ## Student 1 code

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import LinearRegression #Comment: LinearRegression class should be derived from sklearn.linearmodel
from sklearn.cross_validation import cross_val_score # comment: NO cross_validation class is available in sklearn. Rather cross_val_score should be accessed from sklearn.model_selection

# Load data
d = pd.read_csv('../data/train.csv')

# Setup data for prediction
x1 = d.SalaryNormalized # it is not data but it is d
x2 = pd.get_dummies(d.ContractType) # it is not data but it is d

# Setup model
model = LinearRegression()

# Evaluate model
from sklearn.cross_validation import cross_val_score # comment: NO cross_validation class directly is available in sklearn. Rather cross_val_score should be accessed from sklearn.model_selection
from sklearn.cross_validation import train_test_split # comment: NO train_test_split directly class is available in sklearn. Rather cross_val_score should be accessed from sklearn.model_selection
scores = cross_val_score(model, x2, x1, cv=1, scoring='mean_absolute_error')
print(scores.mean())


# ## Most of the comments are given inline. 
# ## Following observation are made:
# ### 1. No exploratory data analysis is done
# ### 2. Trying to predict only using one model...which may not be correct
# ### 3. No model training is carried out
# ### 4. Model fitting is not done
# ### 5. Not able to understand why student is finding the scores of Salary ann ContractType
# ### 6. Student not tried to explore into the categorical data at all...by which be will be leaving out most of the information
# ### 7. Many errors are there in the code
# 
# # It can be judged that student have not got the conceptual clarity on what he has to do analysis because he tring to identify relationship between salary and ContractType. Here ContractType is filled with dummies. So he tring to identifying relationship with dummies

# # Student 2 code review 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score # comment: NO cross_validation class is available in sklearn. Rather cross_val_score should be accessed from sklearn.model_selection

# Load data
data = pd.read_csv('../data/train.csv')


# Setup data for prediction
y = data.SalaryNormalized
X = pd.get_dummies(data.ContractType)

# Setup model
model = LinearRegression()

# Evaluate model
scores = cross_val_score(model, X, y, cv=5, scoring='mean_absolute_error')
print(scores.mean())


# ## Most of the comments are given inline. 
# ## Following observation are made:
# ### 1. No exploratory data analysis is done
# ### 2. Trying to predict only using one model...which may not be correct
# ### 3. No model training is carried out
# ### 4. Model fitting is not done
# ### 5. Not able to understand why student is finding the scores of Salary and ContractType
# ### 6. Student not tried to look explore the categorical data at all...by which be will be leaving out most of the information
# 
# # It can be judged that student have not got the conceptual clarity on what he has to do analysis because he tring to identify relationship between salary and ContractType. Here ContractType is filled with dummies. So he tring to identifying relationship with dummies
