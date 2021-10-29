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


for i in range(train_time):
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
for i in range(trade_time):
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





# In[ ]:





# In[ ]:




