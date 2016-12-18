import numpy as np
import matplotlib.pyplot as plt


#A basic fully connected feedforward neural net with rectangular hidden layers and 
#sigmoid nonlinearity
class basic_neural_net(object):
    def __init__(self, h = 2, npl = 4, in_width = 1, out_width = 1, eta = 0.1):
        #Store params locally
        self.h = h#Number of hidden layers
        self.npl = npl#nodes per hidden layer
        self.in_width = in_width#How many dims is the input?
        self.out_width = out_width#How many dims is the output?
        self.eta = 0.1#Learning rate for sgd
        
        self.L = h + 1#How many total layers?
        
        ###Initialize things
        #bias vector
        self.b = [np.random.normal(size = [self.npl,1]) for i in range(self.h)]
        self.b.append(np.random.normal(size = [self.out_width,1]))
        
        ##Create and intialize w, the weight tensor
        self.w = []
        
        #Input to hidden synapses
        if self.h > 0:
            self.w.append(np.random.normal(size = [npl, in_width]))
        #hidden to hidden synapses
        for i in range(self.h-1):
            self.w.append(np.random.normal(size = [npl, npl]))
        #hidden to output synapses
        self.w.append(np.random.normal(size = [out_width, npl]))
    
    #Compute feedforward activations X should be a pxn numpy array, where p is dimensions and
    #n are observations
    def forward(self, X):
        #Initializer our Z list, the linear combinations of our weights, 
        #and the a list, which contains each activation
        self.a = [X]
        self.z = []
        
        for i in range(self.L):
            self.z.append(np.dot(self.w[i], self.a[i]) + self.b[i])
            self.a.append(self.nonlinearity(self.z[i]))
            
        return(self.a[-1])
    
    #Do backprop. "forward" must have been called first.
    def backprop(self, y):
        ##Get deltas (errors for each layer)
        #Compute output delta
        self.delta = [self.cost(self.a[-1], y, deriv = True) * self.nonlinearity(self.z[-1], deriv = True)]
        
        #Compute deltas for hidden layers
        for i in range(self.h):
            self.delta.append(np.dot(self.w[-(i+1)].T, self.delta[-1]) * self.nonlinearity(self.z[-(i+1)], deriv = True))
        
        ##Do gradient descent
        #On bias units
        #print 'First b\'s'
        for i in range(len(self.b)):
            #print 'Looking at the ' + str(i) + ' th b: ' + str(self.b[i])
            #print 'Looking at the ' + str(-(i+1)) + ' th delta: ' + str(self.delta[-(i+1)])
            self.b[i] = self.b[i] - self.eta * self.delta[-(i+1)]
        
        #print 'now w\'s'
        for i in range(len(self.w)):
            #print 'Looking at the ' + str(i) + ' th w: ' + str(self.w[i])
            #print 'Looking at the ' + str(-(i+1)) + ' th delta: ' + str(self.delta[-(i+1)])
            #print 'Looking at the ' + str(-(i+2)) + ' the activation: ' + str(self.a[-(i+2)])
            self.w[i] = self.w[i] - self.eta * self.delta[-(i+1)] * self.a[-(i+2)]
        #self.b = [self.b[i] - self.eta * self.delta[-(i+1)] for i in range(self.L)]
        #On weight units
        #self.w = [self.w[i] - self.eta * self.delta[-(i+1)] * self.a[-(i+2)] for i in range(self.L)]
        
        
    #Return the cost of guessing est when the truth is tru for one case. If dervi = True, 
    #return the derivative of this.
    #Currently implemented l2 loss
    def cost(self, est, tru, deriv = False):
        if not deriv:
            return(0.5 * np.linalg.norm(tru - est, ord = 2))
        else:
            return(est - tru)
            
    #Return a nonlinear function evaluated at x. If dervi = True, 
    #return the derivative of this.
    #Currently implemented is the sigmoid function
    def nonlinearity(self, z, deriv = False):
        if not deriv:
            return 1 / (1 + np.exp(-z))
        else:
            fz = self.nonlinearity(z,False)
            return(fz * (1.0-fz))    




##Generate some data
n = 100
p = 1
out_p = 1
prop = 0.5
iters = 1000000

count_0 = np.floor(n * prop)
count_1 = n - count_0
X = np.concatenate((np.random.normal(0,2,size=[count_0,1]), np.random.normal(-20,2,size=[count_0,1]), np.random.normal(20,2,size=[count_1,1])))
y = np.concatenate([np.repeat(0,count_0),np.repeat(1,count_1*2)])
plt.hist(X[:count_0])
plt.hist(X[count_0:])
plt.show()



bnn = basic_neural_net(h = 1, npl = 4)


preds = [bnn.forward(np.reshape(X[i], [1,p])) for i in range(n)]
err1 = [bool(bnn.forward(np.reshape(X[i], [1,p])) > 0.5) == bool(y[i]) for i in range(150)]

#Do sgd
for i in range(iters):
    d = i % 100
    #preds[d] = bnn.forward(np.reshape(X[d], [1,p]))
    #err[d] = bool(preds[d] > 0.5) == y[d]
    bnn.forward(np.reshape(X[d], [1,p]))
    bnn.backprop(np.reshape(y[d], [1, out_p]))
    
    
    #if i % 10000000 == 0:
        #print str(np.mean(err))


err = [bool(bnn.forward(np.reshape(X[i], [1,p])) > 0.5) == bool(y[i]) for i in range(150)]

print "From " + str(np.mean(err1)) + " to " + str(np.mean(err))