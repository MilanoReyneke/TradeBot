#!/usr/bin/env python
# coding: utf-8

# pip install luno-python

# In[14]:


from luno_python.client import Client
import time

API_key = input("Enter API key:")
secret_key = input("Enter secret API key:")

c = Client(api_key_id=API_key, api_key_secret=secret_key) #connecting to Luno


#Summarizing pre-trading account balances
balance = c.get_balances()

pre_zar_bal = balance['balance'][5]['balance']
pre_xbt_bal = balance['balance'][1]['balance']
net_value = float(pre_xbt_bal)*(int(float(c.get_ticker('XBTZAR')['bid']))) + int(float(pre_zar_bal))

print("Your pre-trading ZAR balance is: R",pre_zar_bal)    
print("Your pre-trading XBT balance is: BTC",pre_xbt_bal)
print("Your total worth is: R", net_value)


period = int(input("Enter trading period - time between trades being considered in seconds:"))
train_time = 72000/period - 14
trade_time = 14400/period

#Training of neural net with 20 hours of currency price data
Net = NeuralNet()
tickers = []

for i in range(14):
    t_pre = c.get_ticker('XBTZAR')['bid']
    t = int(float(t_pre))/100000                    #to prevent stack overflow
    tickers.append(t)
    time.sleep(period)

last_pre = c.get_ticker('XBTZAR')['bid']
last = int(float(last_pre))/100000
diff = last - tickers[13]
goal = 0
if diff > 0:
    goal = 1
else:
    goal = 0
    
Net.fit(tickers, goal)    


for i in range(10):          #train_time):
    for j in range(13):
        tickers[j] = tickers[j+1]
    t_pre = c.get_ticker('XBTZAR')['bid']
    t = int(float(t_pre))/100000                     #to prevent stack overflow
    tickers[13] = t
    time.sleep(period)
    
    last_pre = c.get_ticker('XBTZAR')['bid']
    last = int(float(last_pre))/100000
    
    diff = last - tickers[13]
    if diff > 0:
        goal = 1
    else:
        goal = 0
    Net.fit(tickers, goal)  
    

    
    
# Trading for 4 hours using the neural net to make price predictions   
for i in range(5):            #trade_time):
    for j in range(13):
        tickers[j] = tickers[j+1]
    t_pre = c.get_ticker('XBTZAR')['bid']
    t = int(float(t_pre))/100000                     #to prevent stack overflow
    tickers[13] = t

    prediction = Net.predict(tickers)
    balance = c.get_balances()

    zar_bal = balance['balance'][5]['balance']
    xbt_bal = balance['balance'][1]['balance']

    if prediction==True and float(zar_bal) >= 500:
        c.post_market_order(pair='XBTZAR', type='BUY', counter_volume = int(float(zar_bal)))
    elif prediction==True and float(zar_bal) <= 500:
        pass
    elif prediction==False and float(xbt_bal) >= 0.0005:
        c.post_market_order(pair='XBTZAR', type='SELL', base_volume = round(xbt_bal, 4))
    elif prediction==False and float(xbt_bal) <= 0.0005:
        pass
    time.sleep(period)


# Summarizing post-trading account balances and profit
balance = c.get_balances()

post_zar_bal = balance['balance'][5]['balance']
post_xbt_bal = balance['balance'][1]['balance']

print("Your post-trading ZAR balance is: R",post_zar_bal)
print("Your post-trading XBT balance is: BTC",post_xbt_bal)

profit=(int(float(c.get_ticker('XBTZAR')['bid']))*float(post_xbt_bal)+int(float(post_zar_bal)))-net_value

print("Your prifit is: R", profit)


# In[13]:


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


# In[ ]:





# In[ ]:




