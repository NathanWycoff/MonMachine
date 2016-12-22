# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:58:57 2016

@author: nathan
"""
import numpy as np
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self, h = 0, npl = 1, in_width = 1, out_width = 1, eta = 0.1, decay = 1):
        #Store params locally
        self.h = h#Number of hidden layers
        self.npl = npl#nodes per hidden layer
        self.in_width = in_width#How many dims is the input?
        self.out_width = out_width#How many dims is the output?
        self.eta = 0.1#Learning rate for sgd
        self.decay = 1.01
        
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
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.w]
        for x, y in mini_batch:
            self.predict(x)
            delta_nabla_b, delta_nabla_w = self.backprop2(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.w = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.w, nabla_w)]
        self.b = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.b, nabla_b)]
    #def swag_out(self,x):
                           
    def predict(self, x):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.b" and "self.w"."""
        self.nabla_b = [np.zeros(b.shape) for b in self.b]
        self.nabla_w = [np.zeros(w.shape) for w in self.w]
        # feedforward
        activation = x
        self.activations = [x] # list to store all the activations, layer by layer
        self.zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.b, self.w):
            z = np.dot(w, activation)+b
            self.zs.append(z)
            activation = sigmoid(z)
            self.activations.append(activation)
        
        return(self.activations[-1])
            
    def backprop2(self,x,y):
        # backward pass
        delta = self.cost_derivative(self.activations[-1], y) * \
            sigmoid_prime(self.zs[-1])
        self.nabla_b[-1] = delta
        self.nabla_w[-1] = np.dot(delta, self.activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.L):
            z = self.zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.w[-l+1].transpose(), delta) * sp
            self.nabla_b[-l] = delta
            self.nabla_w[-l] = np.dot(delta, self.activations[-l-1].transpose())
        
        #Rase eta to decay
        self.eta = pow(self.eta, self.decay)
        
        return (self.nabla_b, self.nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) 
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



#A basic fully connected feedforward neural net with rectangular hidden layers and 
#sigmoid nonlinearity
class basic_neural_net(object):
    def __init__(self, h = 0, npl = 1, in_width = 1, out_width = 1, eta = 0.1):
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
iters = 6000

count_0 = np.floor(n * prop)
count_1 = n - count_0
X = np.concatenate((np.random.normal(0,2,size=[count_0,1]), np.random.normal(-20,2,size=[count_0,1]), np.random.normal(20,2,size=[count_1,1])))
y = np.concatenate([np.repeat(0,count_0),np.repeat(1,count_1*2)])
plt.hist(X[:count_0])
plt.hist(X[count_0:])
plt.show()

np.random.seed(1)
bnn = basic_neural_net(h = 1, npl = 2)
np.random.seed(1)
network = Network(h = 1, npl = 2)
#network.predict(0)
#bnn.forward(0)
i = 2
#network.update_mini_batch(zip(X[i,], [y[i,]]), network.eta)


#Do sgd
for i in range(iters):
    #d = np.random.randint(0,99)
    d = i % 150
    #preds[d] = network.forward(np.reshape(X[d], [1,p]))
    #err[d] = bool(preds[d] > 0.5) == y[d]
    network.update_mini_batch(zip(X, y), network.eta)
    bnn.forward(np.reshape(X[d], [1,p]))
    bnn.backprop(y[d])
    
    #assert bnn.w == network.w

err = np.mean([bool(network.predict(np.reshape(X[i], [1,p])) > 0.5) == bool(y[i]) for i in range(150)])
err1 = np.mean([bool(bnn.forward(np.reshape(X[i], [1,p])) > 0.5) == bool(y[i]) for i in range(150)])