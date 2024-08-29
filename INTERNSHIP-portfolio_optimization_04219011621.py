#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


assets=pd.read_csv(r'C:\Users\kawal\Downloads\USDATA_.csv')
assets.head()


# In[3]:


#assigning weights to the stocks
equal_weight=0.018
weights=np.full(55,equal_weight)
print(weights)


# In[4]:


stocksstartdate='2018-08-18'
stocksenddate='2023-08-17'


# In[5]:


# visually show stock
title='PORTFOLIO CLOSED PRICE HISTORY'
# GET THE STOCKS
my_stocks = assets
#create and plot graph
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c],label=c)
plt.title(title)
plt.xlabel('DATE (DAY WISE)',fontsize= 10)
plt.ylabel('CLOSING PRICE ($)', fontsize=10)
plt.show()


# In[6]:


#show the daily simple return
returns=assets.pct_change()
returns


# In[7]:


#create and show the annualized covariance mtrix
cov_matrix_annual=returns.cov()*252
cov_matrix_annual


# In[8]:


# calculate the portfolio variance 
port_variance=np.dot(weights.T,np.dot(cov_matrix_annual,weights))
port_variance


# In[9]:


# calculate the portfolio volatility aka standard deviation
port_volatility=np.sqrt(port_variance)
port_volatility


# In[10]:


#calculate the annual portfolio return
portfolioSimpleAnnualReturn=np.sum(returns.mean()*weights)*252
portfolioSimpleAnnualReturn


# In[11]:


#show the expected annual return , volatility(risk), and variance
percent_var=str(round(port_variance,2)*100)+'%'
percent_volatility=str(round(port_volatility,2)*100)+ '%'
percent_return=str(round(portfolioSimpleAnnualReturn,2)*100)+ '%'
print("EXPECTED ANNUAL RETURN OF THIS STOCK IS :", percent_return)
print("ANNUAL VOLATILITY :", percent_volatility)
print("ANNUAL VARIANCE :", percent_var)


# In[12]:


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


# In[13]:


#Portfolio OPTIMIZATION

# caluclate the expected returns and the annualized sample covariance matrix of asset return
mu= expected_returns.mean_historical_return(assets)
S= risk_models.sample_cov(assets)

#optimize for max sharp ratio 
ef= EfficientFrontier(mu, S)
weights=ef.max_sharpe()
cleaned_weights=ef.clean_weights()  # any weights whose values is less than some cuttoff value it rounds those stocks to zero
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


# In[14]:


oracle = 0.10461
nvidia=0.33753
microsoft=0.00303
broadcom = 0.36348
church_and_dwight =0.19136
s= 0.10461+0.33753+0.00303+0.36348+0.19136
print(s)


# In[15]:


#get the discrete allocation of each stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices =get_latest_prices(assets)
discreteallocation= DiscreteAllocation(weights,latest_prices,total_portfolio_value=15000)
allocation,leftover=discreteallocation.lp_portfolio()
print("DISCRETE ALLOCATION :", allocation)
print("FUNDS REMAINING : ${:.2f}".format(leftover))


# In[ ]:




