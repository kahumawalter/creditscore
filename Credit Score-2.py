#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras import Sequential
from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.compose import make_column_selector, make_column_transformer


# In[123]:


filename = 'downloads/train-2.csv'
df = pd.read_csv(filename)
df.head()


# In[234]:


#The dataset has no missing values
df.info()


# In[127]:


df['Credit_Score'] = df['Credit_Score'].replace({'Poor':0, 'Standard':1, 'Good':2})
df['Credit_Mix'] = df['Credit_Mix'].replace({'Bad':0, 'Standard':1, 'Good':2})
df.head()


# In[150]:


#Heatmap of the correlation between features.

corr = df.corr()

import seaborn as sns
sns.heatmap(corr, cmap = 'Blues',annot =True)
sns.set(rc = {'figure.figsize':(20,12)})         


# In[151]:


# The above heatmap shows there is a high positive correlation between Credit_score and features like Credit_mix and Credit_history_age
# There is also a high negative correlation between Credit_score and features like Outstanding Debt, Num_credit_inquiries
# Num_of Delayed_Payment, Delay_from_due_date, Num_of_loan, Interest_rate, Num_credit_card and Num_Bank_accounts. 


# In[163]:


sns.boxplot(x='Credit_Score', y='Outstanding_Debt',data=df);
plt.title('Credit Score in Comparison to Outstanding Debt', fontsize=20);


# In[164]:


# The box plot above shows that those with a credit score of 0 (Poor) have an outstanding debt of about 2,000.  
# According to the interquartile range (IQR) it ranges between 1500 and 2500.

# Those with a credit score of 1 (Standard) have an outstanding debt that ranges from 500 to 1500 according to the
#interquartile range. While as those with a credit score of 2 (Good) have an outstanding debt that ranges from 400
# to 1200. 

#However all these classes have outliers with an outstanding debt of 5000. 


# In[166]:


#Using sample data of 1000 to avoid overplotting 
df_sample = df.sample(1000)
sns.relplot(df_sample['Annual_Income'],df_sample['Occupation'], hue = df_sample['Outstanding_Debt'],height=8.27, aspect=11.7/8.27).set(title='Annual Income Versus Outstanding Debt' );


# In[ ]:


# Those who earn over 100,000 annually have an outstanding debt of less than 2,000 while as those who earn less, 
# have outstanding debts of upto 4000.Scientists, lawyers and architect are the only outliers who earned over 
# 175,000 annually


# In[167]:


# Line Plot Visualization 
sns.lineplot(
    x=df['Age'], 
    y=df['Annual_Income']);
sns.set(rc={'figure.figsize':(10,8)}, font_scale=2.5, style='whitegrid')
plt.title('Annual Income According To Age', fontsize = 16);


# In[168]:


# According to the line graph above the lowest income earners are below the age of 20 and majority of those earn 
# less than 40,000 usd annually. The highest earners are between the age of 45 and 52, earning over 50,000 usd 
# annually. There are outliers below the age of 20 who earn more than 50,000 usd annually. 
# For the age group of 50 and above, the lowest earner gets about 52,000 usd annually.


# In[169]:


#Splitting the data 
X = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)


# In[170]:


# Selectors
cat_selector = make_column_selector(dtype_include='object')
num_selector = make_column_selector(dtype_include='number')
# Scaler
scaler = StandardScaler()
# One-hot encoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)


# In[171]:


# Numeric pipeline
numeric_pipe = make_pipeline(scaler)
numeric_pipe


# In[172]:


# Categorical pipeline
categorical_pipe = make_pipeline(ohe)
categorical_pipe


# In[173]:


# Tuples for Column Transformer
number_tuple = (numeric_pipe, num_selector)
category_tuple = (categorical_pipe, cat_selector)
# ColumnTransformer
preprocessor = make_column_transformer(number_tuple, category_tuple)
preprocessor


# In[174]:


preprocessor.fit(X_train)


# In[175]:


# Transform train and test
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# In[176]:


from sklearn.preprocessing import LabelEncoder, LabelBinarizer
# encode labels
encoder = LabelEncoder()
 
encoder.fit(y_train)
y_train_enc = encoder.transform(y_train)
y_test_enc = encoder.transform(y_test)
 
# make a record of the classes, in order of the encoding, in case we want to 
# translate predictions into credit score names later.
classes = encoder.classes_
 
# binarize labels
binarizer = LabelBinarizer()
 
binarizer.fit(y_train_enc)
y_train_bin = binarizer.transform(y_train_enc)
y_test_bin = binarizer.transform(y_test_enc)
 
# check results
print('Original Target')
print(y_train.head())
 
print('\nEncoded Target')
print(y_train_enc[:5])
 
print('\nBinarized Target')
print(y_train_bin[:5])


# In[222]:


# create model architecture
 
# define some parameters
input_dim = X_train_processed.shape[1]
num_classes = len(classes)
 
# instantiate the base model
multi_model = Sequential()
 
# add layers
multi_model.add(Dense(30, input_dim=input_dim, activation='relu'))
multi_model.add(Dropout(.3))
multi_model.add(Dense(30, activation='relu'))
multi_model.add(Dropout(.3))
multi_model.add(Dense(num_classes, activation='softmax'))
multi_model.summary()


# In[223]:


# compiling model with categorical_crossentropy
 
multi_model.compile(loss='categorical_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy',
                             metrics.Precision(),
                             metrics.Recall()])


# In[224]:


# fit model
 
history = multi_model.fit(X_train_processed, y_train_bin,
                          validation_data=(X_test_processed, y_test_bin),
                          epochs=100,
                          verbose=0)


# In[225]:


# Learning history plotting function

def plot_history(history):
  """Takes a keras model learning history and plots each metric"""
  
  metrics = history.history.keys()
  
  for metric in metrics:
      if not 'val' in metric:
        plt.plot(history.history[f'{metric}'], label=f'{metric}')
        if f'val_{metric}' in metrics:
          plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
        plt.legend()
        plt.title(f'{metric}')
        plt.show()


# In[226]:


# plot learning history
plot_history(history)


# In[227]:


# The above graphs show us that the training data isn't far off the testing data with a difference of about 0.1


# In[228]:


# get raw predictions
raw_pred = multi_model.predict(X_test_processed)
 
# display predictions and binarized true labels
print('Raw Predictions\n', raw_pred[:5])
print('\nbinarized y_test\n', y_test_bin[:5])


# In[229]:


# convert predictions and labels into integers representing each credit class.
y_pred = np.argmax(raw_pred, axis=1)
y_true = np.argmax(y_test_bin, axis=1)
 
print('integer predictions', y_pred)
print('integer true labels', y_true)


# In[230]:


# printing classification report and confusion matrix
 
print(classification_report(y_true, y_pred))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                        display_labels=classes);
                                        


# In[231]:


#Checking for classes and balance
y_test.value_counts()


# In[232]:


#The model was able to accurately predict 83% of the credit score. However the model was able to predict the standard class
#better because of the class imbalance in the data provided. 


# In[233]:


#Seeing how the predictions compare to the original values. 

prediction_df = X_test.copy()
prediction_df['True Credit'] = y_true
prediction_df['Predicted Credit'] = y_pred

prediction_df.head()


# In[ ]:




