# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:29:17 2016

@author: Nathan Wycoff

Just making sure my gp funcs work. This file will be deleted when it's confirmed
that it works.
"""

#Import my GP funcs
import sys
sys.path.append('/home/nathan/Documents/Documents/Self Study/MonMachine/')
from gp_optimizer import gp_posterior, get_expected_improvement

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(5)

#Generate some data
n = 50
X = np.linspace(-2,2, n)
y = stats.norm.pdf(X)
plt.plot(X, y)
plt.show()


#Store the function eval
func_eval = stats.norm.pdf

#Get the initial points
r = 3
init_points = [X[i] for i in np.random.choice(range(len(X)),r)]

#Fix storage
train = np.reshape(np.array(init_points), [r, 1])
candidates = np.reshape(np.array([x for x in X if x not in train]), [n-r,1])

#Get initial response
f = [func_eval(x) for x in init_points]

#Get expected improvement
mu, sig = gp_posterior(train, candidates, f)
ei = get_expected_improvement(mu, sig, max(f))

#Plot the GP
plt.plot(candidates, mu)
plt.plot(candidates, mu + sig)
plt.plot(candidates, mu - sig)
plt.plot(train, f, 'ro')
plt.plot(candidates, ei)
plt.show()

#Pick the next f eval.
next_eval = np.argmax(ei)


######################
###################### Begin 1 iter
#Evaluate the new method
train = np.vstack([train, candidates[next_eval,:]])
candidates = np.delete(candidates, next_eval, axis = 0)
new_f = func_eval(candidates[next_eval])
f = np.append(f, new_f)

#Get expected improvement
mu, sig = gp_posterior(train, candidates, f)
ei = get_expected_improvement(mu, sig, max(f))

#Pick the next f eval.
next_eval = np.argmax(ei)

#Plot the GP
plt.plot(candidates, mu)
plt.plot(candidates, mu + sig)
plt.plot(candidates, mu - sig)
plt.plot(train, f, 'ro')
plt.plot(candidates, ei)
plt.show()

###################### End 1 iter
###################### 










##############################################################################
# NEURAL NET STUFF
##############################################################################
ann = learner_1 = neural_q_learner(state_size, action_size = 3, eps_l = 0.1, eps_dyn = 0.9, \
    h = 2, eta = 0.001, max_err = 0.1, h_size = 4, mem_size = 100)

