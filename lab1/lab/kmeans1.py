import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', None)

data = pd.read_csv('Wholesale_customers_data.csv')
#print(data.head())

categorical_features = ['Channel', 'Region']
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

print("Data head: ")
#print(data[continuous_features].describe())
"""
for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
"""
data.drop(categorical_features, axis=1)
print(data.head())
print("Data shape: ", data.shape)
#mms = MinMaxScaler()
#mms.fit(data)
#data_transformed = mms.transform(data)

kmeans_at_5 = KMeans(n_clusters=5)
kmeans_at_5 = kmeans_at_5.fit(data)
# print(kmeans_at_5.labels_)


cluster_map = pd.DataFrame()
cluster_map['data_index'] = data.index.values
cluster_map['cluster'] = kmeans_at_5.labels_
y = kmeans_at_5.labels_
# print(cluster_map[cluster_map.cluster == 3])
# print(cluster_map)

for i in range(5):
    print("No. of items in cluster ", i, ": ", cluster_map[cluster_map.cluster == i]['cluster'].count())

# print(homogeneity_score(cluster_map['cluster']))
Sum_of_squared_distances = []
K = range(2,20)
for k in K:
    km = KMeans(n_clusters=k, max_iter=100)
    km = km.fit(data)
    Sum_of_squared_distances.append(km.inertia_)

print("Elbow measure:")
plt.figure(figsize=(6, 6))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')

plt.figure(figsize=(6, 6))
data_norm = (data - data.min())/(data.max() - data.min())
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(data_norm))
print(transformed)
plt.scatter(transformed[y==0][0], transformed[y==0][1], label='Class 1', c='red')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Class 2', c='blue')
plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Class 3', c='lightgreen')
plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Class 4', c='pink')
plt.scatter(transformed[y==4][0], transformed[y==4][1], label='Class 5', c='orange')
plt.legend()

plt.show()
