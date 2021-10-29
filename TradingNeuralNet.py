#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# A two layered neural network is used that takes 14 points of crypto-currency price data
class NeuralNet():
     
        
    def __init__(self, layers=[14,10,1], learning_rate=0.01, iterations=1000):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.inputData = None
        self.outputGoal = None
                
    def init_weights(self):
        
        np.random.seed(1) 
        self.params['W1'] = np.random.randn(self.layers[1], self.layers[0]) 
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[2],self.layers[1]) 
        self.params['b2'] = np.random.randn(self.layers[2],)
    
    def relu(self,Z):
       
        return np.maximum(0,Z)

    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0]  = 1
        return x


    def sigmoid(self,Z):
        
        return 1/(1+np.exp(-Z))
    
    
    def lossfunc(self,y, yhat):
    
        # a difference squared loss function is used for this net
        lossf = (y-yhat)**2
        
    
        return lossf

    def forward_propagation(self):

        
        Z1 = np.dot(self.params['W1'], self.inputData ) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = np.dot(self.params['W2'], A1) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.lossfunc(self.outputGoal,yhat)

        # save calculated parameters     
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat,loss

    def back_propagation(self,yhat):
        
        # derivatives from the loss function calculated
        y_inv = 1 - self.outputGoal
        yhat_inv = 1 - yhat

        
        dl_wrt_yhat = -2*(self.outputGoal-yhat)
        dl_wrt_sig = yhat*yhat_inv
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2*self.params['W2']
        dl_wrt_w2 = dl_wrt_z2*self.params['A1']
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = np.dot(dl_wrt_A1, self.dRelu(self.params['Z1']))
        dl_wrt_w1 = dl_wrt_z1*self.inputData
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        #update the weights and bias

        
        self.params['W1'] = self.params['W1'] - (self.learning_rate * dl_wrt_w1)
        self.params['W2'] = self.params['W2'] - (self.learning_rate * dl_wrt_w2)
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        # trains the neural net with input data, X, and goal output, y 
        self.inputData = X
        self.outputGoal = y
        self.init_weights() #initialize weights and bias


        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        #neural net's prediction based on input, X
        Z1 = np.dot(self.params['W1'], X) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = np.dot(self.params['W2'], A1) + self.params['b2']
        p = self.sigmoid(Z2)
        if p > 0.5:
            pred = True
        else:
            pred = False
        return pred 

