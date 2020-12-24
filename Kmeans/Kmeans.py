import numpy as np 


class Kmeans():
    """
        Docstring.

        Kmeans Algorithm.

        Parameters:
        K(int): Number of groups to be found
        random_start(int): Number of random starts

        Returns:
        centers(Array): K centers found
        groups(Array): Group of each data point

    """

    @staticmethod
    def costFunction(X, centers, groups):
        m = len(X)
        cost = 0
        for i in range(m):
            cost += Kmeans.euclianDistance(X[i,:].reshape((1,-1)),centers[groups[i],:].reshape((1,-1)))[0][0]

        return cost/m


    @staticmethod
    def euclianDistance(a,b):
        distances = np.zeros((len(a),len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                distances[i][j] = np.linalg.norm(a[i,:]-b[j,:],axis = 0)

        return distances




    def __init__(self,K, random_start = 1, epochs = 100):
        self.K = K
        self.random_start = random_start
        self.epoch = epochs
        self.centers = 0
        self.groups = 0

    
    def fit(self,X):
        costs = []
        groupsList = []
        centersList = []

        for nSt in range(self.random_start):
            #Random initializing centers
            initmean = np.random.choice(range(len(X)), size = self.K,replace = False)
            centers = X[initmean]
            for e in range(self.epoch):
                # print(f"Iteration: {e}")

                distances = Kmeans.euclianDistance(X,centers)
                groups = np.argwhere((distances == distances.min(axis = 1,keepdims = True)))[:,1]

                #Update centers with mean
                for i in range(self.K):
                    centers[i,:] = (X[groups == i]).mean(axis = 0)


            cost = Kmeans.costFunction(X, centers, groups)

            costs.append(cost)
            groupsList.append(groups)
            centersList.append(centers)
            print(f"Start{nSt} - Cost: {cost} - Best {min(costs)}")

        self.centers = centersList[costs.index(min(costs))]
        self.groups = groupsList[costs.index(min(costs))]
            

