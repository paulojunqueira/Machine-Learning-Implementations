import numpy as np 

class LinearRegressionMulti():
    """ Multivariate Linear Regression model 
    """
    def __init__(self,X, y):
        self.y = y
        self.m , self.n = X.shape
        self.x = np.hstack([np.ones((self.m,1)),x])
        

    def fit(self, epoch, eta):

        thetas = self.initialize()

        cost = []
        for i in range(epoch):
            print(f'Epoch: {i}/{epoch}')
            
            h = (np.dot(thetas.T, self.x.T)).reshape((self.m,1))

            #Cost Function calulation
            cost.append(self.costFunction(h,y))

            #Gradient Descent
            thetas = thetas - (eta/self.m)*np.dot((h-y).T,self.x).T


        return cost, thetas

    def costFunction(self, h, y):
        #Calculates Costs
       j = (1/(2*self.m))*np.sum(np.power((h-y),2))
       return j

    
    def initialize(self):
        #Random Initialization of  Parameters
        theta = np.random.randn(self.n+1,1)

        return theta



import matplotlib.pyplot as plt

#Example Sample

x = np.arange(0,200).reshape((100,2))
y = np.random.randn(100,1)*15 + 0.5


model = LinearRegressionMulti(x,y)
t = model.initialize()
cost, params = model.fit(100, 0.00001)
plt.plot(cost)

x = np.hstack([np.ones((len(x),1)),x])

yhat = (np.dot(params.T, x.T)).reshape((len(x),1))


