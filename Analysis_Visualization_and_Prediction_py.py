"""
Created on Thu Jul 16 13:15:39 2020

@author: Victoria Adedoyin Adeoye
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

case_study= pd.ExcelFile("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx")
emp_left= case_study.parse("Employees who have left")
emp_exist= case_study.parse("Existing employees")
emp_left.head()

emp_exist.head()

emp_left["left"]= 1
emp_left
emp_exist["left"]= 0
emp_exist

#combining both sheets
combined_data = pd.concat([emp_left, emp_exist])
combined_data.head(10)
combined_data.isnull()

combined_data.describe()

#checking the averages of the features, grouping by "left"
left = combined_data.groupby('left')
comparison= left.mean()

#visualizing the difference between features of emp_left and emp_exist
features=['number_project','time_spend_company','promotion_last_5years','Work_accident','dept','salary']
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = combined_data, hue='left')
    plt.xticks(rotation=90)
    plt.title("Comparing emp_left and emp_exist")
    plt.savefig("Comparison_viz.png",dpi=1000)
    
# feature selection
X = combined_data.iloc[:, 1:10].values
Y = combined_data.iloc[:, 10].values

# encoding categorical data
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
X[:,7]=le.fit_transform(X[:,7])
X[:,8]=le.fit_transform(X[:,8])

# splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)

#fitting random forest classifier into test set
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=30,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

# predicting test results
Y_pred= classifier.predict(X_test)

# evaluating model performance
from sklearn import metrics
print(metrics.accuracy_score(Y_test, Y_pred))
























