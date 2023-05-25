#load and undestand data
import pandas as pd
ds=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\19\faithful.csv")
print(ds.head(5))
print(ds.tail(5))
print(ds.shape)
print(ds.describe())

# StandardScaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_std=sc.fit_transform(ds)

#kmean
from sklearn.cluster import KMeans
km=KMeans(n_clusters=2,max_iter=100)
km.fit(x_std)

#center
centroids=km.cluster_centers_
print(centroids)

#charts

import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,8))
plt.scatter(x_std[km.labels_==0,0],x_std[km.labels_==0,1],c="green",label="cluster1")
plt.scatter(x_std[km.labels_==1,0],x_std[km.labels_==1,1],c="blue",label="cluster2")
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()