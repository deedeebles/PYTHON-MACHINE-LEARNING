# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 06:15:19 2020

@author: Victori Adedoyin Adeoye
"""


import pandas as pd
import matplotlib.pyplot as plt

case_study= pd.ExcelFile("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx")
emp_exist= case_study.parse("Existing employees")
emp_exist.head()

X= emp_exist.iloc[:,1:3].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.savefig("Number_of_Clusters_for_emp_exist.jpg",dpi=500)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[Y_kmeans==0, 0], X[Y_kmeans==0, 1], s=100, c='purple', label= 'Category 1')
plt.scatter(X[Y_kmeans==1, 0], X[Y_kmeans==1, 1], s=100, c='blue', label= 'Category 2')
plt.scatter(X[Y_kmeans==2, 0], X[Y_kmeans==2, 1], s=100, c='green', label= 'Category 3')
plt.scatter(X[Y_kmeans==3, 0], X[Y_kmeans==3, 1], s=100, c='cyan', label= 'Category 4')
plt.scatter(X[Y_kmeans==4, 0], X[Y_kmeans==4, 1], s=100, c='orange', label= 'Category 5')
plt.title('Categories of Existing Employees')
plt.xlabel('satisfaction_level')
plt.ylabel('last_evaluation')
plt.legend()
plt.savefig("Types_of_existing_employees.png",dpi=500)
plt.show()