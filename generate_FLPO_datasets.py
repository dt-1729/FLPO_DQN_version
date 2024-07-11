import numpy as np
from numpy import *;
import matplotlib.pyplot as plt; plt.rcParams.update(plt.rcParamsDefault);
import pickle;

def generate_dataset(n, cov, k,scale=30,seed=0):
    # Generate random means for each cluster
    np.random.seed(seed)
    means = np.random.rand(k, 2)*scale

    # Generate random points in k clusters
    nodes = np.zeros((n, 2))
    for i in range(n):
        cluster = np.random.randint(0, k)  # Choose a random cluster
        nodes[i] = np.random.multivariate_normal(means[cluster], cov)
    Y_s = np.tile(np.sum(nodes, axis=0), (k,1))/n
    dest = np.random.rand(1,2)*scale
    return nodes,Y_s,dest

# Example usage 
n = 500  # Total number of points
var = 5e-1 # variance of each cluster
cov = np.eye(2) * var # Covariance matrix
k = 20  # Total number of clusters
nodes, Y_s, dest = generate_dataset(n, cov, k, scale=40,seed = 0)
plt.scatter(nodes[:,0],nodes[:,1],marker='.')
plt.scatter(Y_s[:,0],Y_s[:,1],marker='x')
plt.scatter(dest[:,0],dest[:,1],marker='^')
plt.grid()
plt.legend(['node locations','initial facility locations','destination'])
plt.show()
smallCellNetdata = {'numNodes':n, 'numFacilities':k, 'nodeLocations':nodes, 'facilityLocations':Y_s, 'destinationLocation':dest}
filename = 'smallCellNetData.pkl';

with open(filename, 'wb') as file:
    pickle.dump(smallCellNetdata, file)