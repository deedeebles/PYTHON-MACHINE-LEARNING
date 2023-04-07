# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 02:10:31 2020

@author: Victoria Adedoyin Adeoye
"""


import pandas as pd
import matplotlib.pyplot as plt

case_study= pd.ExcelFile("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx")
emp_left= case_study.parse("Employees who have left")
emp_left.mean()

X= emp_left.iloc[:,1:3].values

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
plt.savefig("Number_of_Clusters_of_employees_who_left.jpg", dpi=500)
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[Y_kmeans==0, 0], X[Y_kmeans==0, 1], s=100, c='purple', label= 'Category 1')
plt.scatter(X[Y_kmeans==1, 0], X[Y_kmeans==1, 1], s=100, c='blue', label= 'Category 2')
plt.scatter(X[Y_kmeans==2, 0], X[Y_kmeans==2, 1], s=100, c='green', label= 'Category 3')
plt.title('Categories of Employees who have left')
plt.xlabel('satisfaction_level')
plt.ylabel('last_evaluation')
plt.legend()
plt.savefig("Categories_of_Employees_who_have_left.png",dpi=500)
plt.show()



