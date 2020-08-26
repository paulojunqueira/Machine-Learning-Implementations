import numpy as np 

class LinearRegression():
    """ Univariate Linear Regression model 

    """
    def __init__(self,x, y):
        self.x = x
        self.y = y
        self.m = x.shape[0]
    
    def fit(self, epoch, eta):

        params = self.initialize()
        cost = []
        for i in range(epoch):
            print(f'Epoch: {i}/{epoch}')
            
            h = params['theta0'] + params['thetas']*x

            #Cost Function calulation
            cost.append(self.costFunction(h,y))

            #Gradient Descent
            temp0= params['theta0'] - (eta/self.m)*(np.sum(h - y))
            temp1 = params['thetas'] - (eta/self.m)*(np.sum((h - y)*x))

            #Update Parameters
            params['theta0'] = temp0
            params['thetas'] = temp1

        return cost, params

    def costFunction(self, h, y):
        #Calculates Costs
       j = (1/(2*self.m))*np.sum(np.power((h-y),2))
       return j

    
    def initialize(self):
        #Initialize Parameters
        params = {}
        params['theta0'] = np.random.rand()
        params['thetas'] = np.random.rand()

        return params


import matplotlib.pyplot as plt

#Example Sample

x = np.arange(0,100).reshape((100,1))
y = np.random.randn(100,1)*15 + x*0.5

#Create Model
model = LinearRegression(x,y)
cost,params = model.fit(100, 0.0001)

plt.figure()
plt.plot(cost)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.grid(True)

yhat = params['theta0'] + params['thetas']*x

plt.figure()
plt.scatter(x, y)
plt.plot(yhat, c = 'red')
plt.ylabel('y')
plt.xlabel('x')

