import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print(plt.style.available)
plt.style.use('seaborn-v0_8')

dfx = pd.read_csv('./csv/xdata.csv')
dfy = pd.read_csv('./csv/ydata.csv')

print(dfx.shape)

#convert into numpy

x = dfx.values
y = dfy.values

print(x)
x = x[:,1:]
y = y[:,1:].reshape(-1,)
print(y)
print(x)

#ploting

query = np.array([2,3]) #creating query
plt.scatter(query[0], query[1],c='red')
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()

#prediction for query point
#knn code

#1. find k nearest neighouburs base on distance
#distance formula between two numpy array - euclidean distance

def distance(x1,x2):
    return np.sqrt(sum(x1-x2)**2)

#apply knn algo
def knn(x,y,querypoint,k=5):
   #pick k nearest neighours

   vals = []

   #for every point in x
   for i in range(x.shape[0]):

    #compute distance
    d = distance(querypoint, x[i])
    vals.append((d,y[i]))

    print(vals)
#2 sort the array
    vals = sorted(vals)
    vals = vals[:k]

#majority vote

    vals = np.array(vals)
    new_values = np.unique(vals[:, 1],return_counts=True)

#index of the max count

    index = new_values[1].argmax()

#map this index with data
    pred = new_values[0][index]
    return pred



knn(x,y,[0,1])