
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

file_path = '/content/Mall_Customers (1).csv'
dataset = pd.read_csv(file_path)

print(dataset.head(10))
print(dataset.shape)
print(dataset.info())

X = dataset.iloc[:, [3, 4]].values

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(scaled_features)

dataset['Cluster'] = y_kmeans

plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[y_kmeans == 0, 0], scaled_features[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1: High income, low spenders')
plt.scatter(scaled_features[y_kmeans == 1, 0], scaled_features[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2: Average income, average spenders')
plt.scatter(scaled_features[y_kmeans == 2, 0], scaled_features[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3: High income, high spenders')
plt.scatter(scaled_features[y_kmeans == 3, 0], scaled_features[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4: Low income, high spenders')
plt.scatter(scaled_features[y_kmeans == 4, 0], scaled_features[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5: Low income, low spenders')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
