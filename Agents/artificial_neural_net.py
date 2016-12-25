# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 22:13:10 2016

@author: Nathan Wycoff
A Multi-layer perceptron neural net. Inspired by http://deeplearning.net/tutorial/mlp.html
and some code portions are copied directly.

Uses Theano to make everything easy and beautiful.

has the ability to merge input dimensions and later split them, as in 
http://papers.nips.cc/paper/531-node-splitting-a-constructive-algorithm-for-feed-forward-neural-networks.pdf
"""

#Theano is a numerical computation library
import theano 
import theano.tensor as T

#Numpy is as well.
import numpy as np

class artificial_neural_net(object):
    """
    ANN/Multilayer Perceptron. A set of Layers and functions to train them using Theano
    
    In particular, the class solves regression problems and uses logistic nonlinearity,
    but it should be simple to modify it to do classification or use other activations funcs.
    
    This class can merge certain input features, and split them later in training.
    See: http://papers.nips.cc/paper/531-node-splitting-a-constructive-algorithm-for-feed-forward-neural-networks.pdf
    , except we pick when they are to be merged; it isn't inferred, and we start
    all merged roots at their parents value (no s.d. shift).
    """
    def __init__(self, in_size, out_size, h, h_size, eta = 0.01, max_err = 10.0, to_merge = [], rng = None):
        """
        :type in_size: int
        :param len_in: Dimensionality of input space BEFORE ANY MERGING
        
        :type out_size: int
        :param len_out: Dimensionality of output space
        
        :type h: uint
        :param h: Number of hidden layers
        
        :type h_size: uint
        :param h_size: Nodes per hidden layer
        
        :type eta: double
        :param eta: non-negative learning rate; premultiplies gradient in sgd
        
        :type max_err: double
        :param max_err: max_err is the absolute maximum error gradient; non-neg
        
        :type to_merge: list
        :param to_merge: list of lists containing input dimensions which should
        be merged to begin with. 
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights. If None, will make one
        """
        #Maintain refernce to rng, make one if not provided
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        
        #Create symbolic vars
        X_var = T.dmatrix('X')#Represents input
        y_var = T.dmatrix('y')#Represents output
        g_var = T.wscalar('g')#Picks one output to do SGD wrt
        
        ##Create and store the layers
        self.layers = []#Storage for layers
        dim_red = len([item for sublist in to_merge for item in sublist])
        starting_in_size = in_size + len(to_merge) - dim_red#Get in_size after merges

        if starting_in_size < 0:
            raise ValueError("to_merge is not consistent with in_size")
            
        len_in = starting_in_size#The input of the first layer is the dim of the input space
        ID = 0#ID for each layer
        
        #Hidden layers
        in_vec = X_var#Start the pipeline
        for i in range(h):
            #Create a layer and append it.
            self.layers.append(Layer(rng, in_vec, len_in, h_size, ID, T.nnet.nnet.sigmoid))
            
            #From 1st iter on, input will be hidden size.
            len_in = h_size
            
            #Append the output of the last layer to the calculation
            in_vec = self.layers[-1].output
            
            #Increment ID
            ID += 1
        
        #Output layer
        self.layers.append(Layer(rng, in_vec, len_in, out_size, '-1'))
        
        #Use this function to predict vals, just need to preprocess the X.
        predict = theano.function([X_var], self.layers[-1].output)
        self.predict = lambda x: predict(self._get_merged_des_mat(x, self.current_merge))
        
        ##Prepare cost function for SGD
        cost = T.mean(T.square(self.layers[-1].output[:, g_var] - y_var))
        grads = [T.jacobian(cost, layer.w) for layer in self.layers]
        
        #What's going to change when we do sgd, and how?
        updates = [
            (layer.w, layer.w - T.clip(eta * grad, -max_err, max_err))
                for grad, layer in zip(grads, self.layers)
            ]
            
        self._sgd = theano.function(inputs = [X_var, y_var, g_var], updates = updates)
        
        #Which dims should we merge?
        self.to_merge = to_merge
        self.current_merge = to_merge[:]
        self.prev_split = []#Contains the merges we've split
        
        
    def grad_on(self, x, y, g):
        """
        Does sgd on target and updates params
        
        These params MUST be matrices (2D numpy arrays), even if the space is
        one dimensional, should be [n,1], not [n,].
        
        :type x: Input matrix (regressors)
        :type y: output matrix (response)
        """
        
        #If there's nothing to merge, don't even bother putting it through the function
        if len(self.to_merge) > 0:
            self._sgd(self._get_merged_des_mat(x, self.current_merge), y, g)
        else:
            self._sgd(x, y, g)
    
    def _get_merged_des_mat(self, X, to_merge = []):
        """
        Sums columns of the design matrix together as specified to create a smaller
        design matrix. The purpose of this is to get around Theano inability to 
        take gradients wrt a function involving T.repeats with unknown repeats.
        
        Returns an np.array of reduced dim. if to_merge is an empty list, simply
        returns X. The merged columns are added to the end.
        
        Meant for internal use.
        
        :type to_merge: list
        :param to_merge: list of lists, each sublist contains columns to combine.
        """
        
        merged_inds = [item for sublist in to_merge for item in sublist]#Unlist everything
        
        #Check that there are no duplicates
        if len(merged_inds) != len(set(merged_inds)):
            print "ERR in get_merged_des_mat: Duplicate Columns Merged"
            return(0)
        
        #Get a list of ALL the merges, including when a column is only be merged with itself.
        not_to_merge = list(set(range(np.shape(X)[1])).difference(merged_inds))
        merged = [[x] for x in not_to_merge] + to_merge
        
        #Merge every column as specified above.
        return(np.array([sum([X[:,j] for j in l]) for l in merged]).T)
    
    def _de_merge_w(self, w, to_merge, current_merge, which_split, previously_split = []):
        """
        Calculates the new weight vector when formerly merged columns are split.
        
        Meant for internal use.
        
        TODO: Make more readable/simplify code, surely there's a better way to write this.
        
        :type w: np 1d array
        :param w: weight vector to be split
        
        :type to_merge: list
        :param to_merge: the merge configuration BEFORE any splits; original to_merge
        
        :type which_split: int
        :param which_split: index of merger in to_merge which is being split.
        
        :type previously_split: list
        :param previously_split: list of integer indexes of places previously split
        """
        
        #Create a new merge list for use in this function, basically takes care of
        #interference from previous splits.
        orig = to_merge[which_split]
        #current_merge = to_merge[:]
        #to_remove = list(np.sort(previously_split))
        #to_remove.reverse()
        #[current_merge.remove(current_merge[i]) for i in to_remove]
        which_split = current_merge.index(orig)
        
        fixed_cols = []
        for col_i in range(np.shape(w)[1]):
            #Get the column we'll be operating on
            w_i = w[:,col_i]
            
            #Get the w of interest, its index, and remove it from the array.
            w_ind = -(len(current_merge) - which_split)#Get the index of the w we want to split
            w_val = w_i[w_ind]#Get the w that we want to split.
            new_cols = current_merge[which_split]#Figure out which cols to add
            w_ret = np.delete(w_i, w_ind)
            
            #Get a flat list of all the other merged columns
            flat_merged = [item for i, sublist in enumerate(current_merge) for item in sublist if i != which_split]
            
            #Get the indices of the weight copies, adjusted for any other mergers and for earlier additions
            new_inds = [col - sum([x < col for x in flat_merged]) - i for i, col in enumerate(new_cols)]
            w_new = np.insert(w_ret, new_inds, w_val)
            
            fixed_cols.append(w_new)
        
        w_ret = np.array(fixed_cols).T
        
        return(w_ret)
        
    def split(self, target):
        """
        De-merges the target merger, copying its weight to the columns which 
        used to compose it.
        
        :type target: uint
        :param target: Index of the "to_merge" list in the "to_merge" object
        passed to the class constructor to be split. Refers to ORIGINAL position.
        """
        
        self.layers[0].set_w(self._de_merge_w(self.layers[0].w.get_value(), self.to_merge, self.current_merge, target, self.prev_split))
        self.current_merge.remove(self.to_merge[target])
        self.prev_split.append(target)
        
    def clone(self):
        """
        Return a new artificial_neural_net object with the same properties as this one, including rng state.
        """
        #Create a new ann object
        ret_rnn = artificial_neural_net(in_size = 0, out_size = 0, h = len(self.layers)-1, h_size= 0, rng = self.rng)
        
        #Give it our weights
        [ret_rnn.layers[i].set_w(self.layers[i].w.get_value()) for i in range(len(self.layers))]
        
        #Update its merger status
        ret_rnn.to_merge = self.to_merge[:]
        ret_rnn.current_merge = self.current_merge[:]
        ret_rnn.prev_split = self.prev_split[:]
        
        return(ret_rnn)
        
        

class Layer(object):
    """
    Respresents one layer of a neural net, based off the deeplearning tutorial 
    at http://deeplearning.net/tutorial/mlp.html
    """
    def __init__(self, rng, in_vec, len_in, len_out, ID = '0', activation = lambda x: x):
        """
        :type in_vec: theano.tensor.TensorType
        :param in_vec: symbolic variable that describes the input of the
        architecture
        
        :type len_in: int
        :param len_in: Represents dimensionality of input space of this layer
        
        :type len_out: int
        :param len_out: Represents dimensionality of output space of this layer
        
        :type ID: str
        :param ID: ID of this layer, for debugging purposes. Set to 0 for input, and '-1'
        for output layer (not required).
        
        :type activation: function
        :param activation: Nonlinearity to be applied at this layer. Default is 
        identity.
        """
        
        #Initialize this layer's weights
        self.w = theano.shared(np.asarray(rng.normal(size = [len_in, len_out])), 'w' + str(ID))
        self.z = T.dot(in_vec, self.w)
        self.output = activation(self.z)
        
        #Store dims/ID
        self.ID = str(ID)
        self.len_in = len_in
        self.len_out = len_out
        
    def set_w(self, new_val):
        """
        Set the weight vector of this layer to some new matrix intelligently. 
        Stores the new dims so they are printed to the interpreter if this layer
        is fed to it.
        
        Does NOT check that the new value agrees with the next layer.
        
        :type new_val: numpy.ndarray
        :param new_val: new matrix (2d np.array) that the weight vector should take.
        """
        self.len_in = np.shape(new_val)[0]
        self.len_out = np.shape(new_val)[1]
        self.w.set_value(new_val)
        
        
    def __str__(self):
        layer_type = 'Input' if self.ID == '0' else 'Hidden'
        layer_type = 'Output' if self.ID == '-1' else layer_type
        return(layer_type + "layer with ID " + str(self.ID) + " of dim " + str(self.len_in) + "x" + str(self.len_out))
        
    __repr__ = __str__