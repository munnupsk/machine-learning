#predicting the spending score based on their annual income based on Mall_Customers.csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#using the elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlable('Number of cluster')
plt.ylabel('wcss')
plt.show()

#applying k means to mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='clusteer 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='clusteer 1')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='clusteer 1')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='clusteer 1')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='clusteer 1')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroids')
plt.title('clusters of clients')
plt.xlabel('Annual income(k$)')
plt.ylabel('spending score(1-100)')
plt.show()
