# Importing libraries
import pandas as pd
import numpy as np
import math
import operator

#### Start of STEP 1

# Importing data
data = pd.read_csv("knn-wisc-bc-data.csv")

#### End of STEP 1
data.head()


# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


# Defining our KNN model
def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}

    length = testInstance.shape[1]

    #### Start of STEP 3
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        #### Start of STEP 3.1
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]
        #### End of STEP 3.1

    #### Start of STEP 3.2
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    #### End of STEP 3.2

    neighbors = []

    #### Start of STEP 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #### End of STEP 3.3
    classVotes = {}

    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    #### Start of STEP 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return (sortedVotes[0][0], neighbors)
    #### End of STEP 3.5


# Creating a dummy testset
testSet = [[87139402, 'B', 12.32, 12.39, 78.85, 464.1, 0.1028, 0.06981,	0.03987, 0.037,	0.1959,	0.05955,
            0.236, 0.6656, 1.67, 17.43,	0.008045, 0.0118, 0.01683, 0.01241,	0.01924, 0.002248, 13.5, 15.64,
            86.97, 549.1, 0.1385, 0.1266, 0.1242, 0.09391, 0.2827, 0.06771]]

test = pd.DataFrame(testSet)

#### Start of STEP 2
# Setting number of neighbors = 1
k = 1
#### End of STEP 2
# Running KNN model
result,neigh = knn(data, test, k)

# Predicted class
print(result)

# Nearest neighbor
print(neigh)

# Setting number of neighbors = 3
k = 3
# Running KNN model
result,neigh = knn(data, test, k)
# Predicted class
print(result)
# Nearest neighbor
print(neigh)

# Setting number of neighbors = 5
k = 5
# Running KNN model
result,neigh = knn(data, test, k)
# Predicted class
print(result)
# Nearest neighbor
print(neigh)

# From Scikit
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data.iloc[:,0:4], data['Name'])

# Predicted class
print(neigh.predict(test))
# 3 nearest neighbors
print(neigh.kneighbors(test)[1])
