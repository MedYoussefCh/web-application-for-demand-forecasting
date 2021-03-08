#!/usr/bin/env python
# coding: utf-8

# In[11]:


#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn        import linear_model
from scipy.optimize import fmin_l_bfgs_b


#---------------------------------------------------------------------------

sdata = open('sampledata.csv')
tsA = sdata.read().split('\n')
tsA = list(map(int, tsA))

#-----------------------------------------------------------------------------------

def holtWinters(ts, p, sp, ahead, mtype, alpha = None, beta = None, gamma = None):

    a, b, s = _initValues(mtype, ts, p, sp)

    if alpha == None or beta == None or gamma == None:
        ituning   = [0.1, 0.1, 0.1]
        ibounds   = [(0,1), (0,1), (0,1)]
        optimized = fmin_l_bfgs_b(_MSD, ituning, args = (mtype, ts, p, a, b, s[:]), bounds = ibounds, approx_grad = True)
        alpha, beta, gamma = optimized[0]

    MSD, params, smoothed = _expSmooth(mtype, ts, p, a, b, s[:], alpha, beta, gamma)
    predicted = _predictValues(mtype, p, ahead, params)

    return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'MSD': MSD, 'params': params, 'smoothed': smoothed, 'predicted': predicted}

def _initValues(mtype, ts, p, sp):
    '''subroutine to calculate initial parameter values (a, b, s) based on a fixed number of starting periods'''

    initSeries = pd.Series(ts[:p*sp])

    if mtype == 'additive':
        rawSeason  = initSeries - initSeries.rolling(min_periods = p, window = p, center = True).mean()
        initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
        initSeason = pd.Series(initSeason) - np.mean(initSeason)
        deSeasoned = [initSeries[v] - initSeason[v % p] for v in range(len(initSeries))]
    else:
        rawSeason  = initSeries / initSeries.rolling(min_periods = p, window = p, center = True).mean()
        initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
        initSeason = pd.Series(initSeason) / math.pow(np.prod(np.array(initSeason)), 1/p)
        deSeasoned = [initSeries[v] / initSeason[v % p] for v in range(len(initSeries))]

    lm = linear_model.LinearRegression()
    lm.fit(pd.DataFrame({'time': [t+1 for t in range(len(initSeries))]}), pd.Series(deSeasoned))
    return float(lm.intercept_), float(lm.coef_), list(initSeason)

def _MSD(tuning, *args):
    '''BFGS optimization to determine the optimal (alpha, beta, gamma) values'''

    predicted = []
    mtype     = args[0]
    ts, p     = args[1:3]
    Lt1, Tt1  = args[3:5]
    St1       = args[5][:]
    alpha, beta, gamma = tuning[:]

    for t in range(len(ts)):

        if mtype == 'additive':
            Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] - Lt)         + (1 - gamma) * (St1[t % p])
            predicted.append(Lt1 + Tt1 + St1[t % p])
        else:
            Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] / Lt)         + (1 - gamma) * (St1[t % p])
            predicted.append((Lt1 + Tt1) * St1[t % p])

        Lt1, Tt1, St1[t % p] = Lt, Tt, St

    return sum([(ts[t] - predicted[t])**2 for t in range(len(predicted))])/len(predicted)

def _expSmooth(mtype, ts, p, a, b, s, alpha, beta, gamma):
    '''calculate the retrospective smoothed values and final parameter values for prediction'''

    smoothed = []
    Lt1, Tt1, St1 = a, b, s[:]

    for t in range(len(ts)):

        if mtype == 'additive':
            Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] - Lt)         + (1 - gamma) * (St1[t % p])
            smoothed.append(Lt1 + Tt1 + St1[t % p])
        else:
            Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] / Lt)         + (1 - gamma) * (St1[t % p])
            smoothed.append((Lt1 + Tt1) * St1[t % p])

        Lt1, Tt1, St1[t % p] = Lt, Tt, St

    MSD = sum([(ts[t] - smoothed[t])**2 for t in range(len(smoothed))])/len(smoothed)
    return MSD, (Lt1, Tt1, St1), smoothed

def _predictValues(mtype, p, ahead, params):
    '''generate predicted values ahead periods into the future'''

    Lt, Tt, St = params
    if mtype == 'additive':
        return [Lt + (t+1)*Tt + St[t % p] for t in range(ahead)]
    else:
        return [(Lt + (t+1)*Tt) * St[t % p] for t in range(ahead)]



results = holtWinters(tsA, 12, 4, 24, mtype = 'additive')
results = holtWinters(tsA, 12, 4, 24, mtype = 'multiplicative')

plt.plot(tsA, color='g', label='series')
plt.plot(results['smoothed'], color='r', label='result')
plt.title('prevision')
plt.xlabel('period')
plt.ylabel('demand')
plt.legend(loc='best')
plt.show()

print("TUNING: ", results['alpha'], results['beta'], results['gamma'], results['MSD'])
print("FINAL PARAMETERS: ", results['params'])
print("PREDICTED VALUES: ", results['predicted'])


# In[ ]:





# In[ ]:




