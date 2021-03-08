# Functions


# Simple And Double Imports
import matplotlib.pyplot as plt

# Tripe Imports
import math
import numpy as np
import pandas as pd

from sklearn import linear_model
from scipy.optimize import fmin_l_bfgs_b

# Arima
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# LSTM


# Functions

# Simple


def single_exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    result.append(alpha * series[n] + (1 - alpha) * result[n])
    # plt.plot(series, color='g', label='series')
    # plt.plot(result, color='r', label='result')
    # plt.title('prevision')
    # plt.xlabel('period')
    # plt.ylabel('demand')
    # plt.legend(loc='best')
    # plt.show()
    return result[:len(series)]


# Double
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+2):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)

    # plt.plot(series, color='g', label='series')
    # plt.plot(result, color='r', label='result')
    # plt.title('prevision')
    # plt.xlabel('period')
    # plt.ylabel('demand')
    # plt.legend(loc='best')
    # plt.show()
    return result[:len(series)]

# Triple


def holtWinters(ts, p, sp, ahead, mtype, alpha, beta, gamma):

    a, b, s = _initValues(mtype, ts, p, sp)

    # if alpha == None or beta == None or gamma == None:
    # ituning   = [0.1, 0.1, 0.1]
    # ibounds   = [(0,1), (0,1), (0,1)]
    # optimized = fmin_l_bfgs_b(_MSD, ituning, args = (mtype, ts, p, a, b, s[:]), bounds = ibounds, approx_grad = True)
    # alpha, beta, gamma = optimized[0]

    MSD, params, smoothed = _expSmooth(
        mtype, ts, p, a, b, s[:], alpha, beta, gamma)
    predicted = _predictValues(mtype, p, ahead, params)

    return {'MSD': MSD, 'smoothed': smoothed}


def _initValues(mtype, ts, p, sp):
    '''subroutine to calculate initial parameter values (a, b, s) based on a fixed number of starting periods'''

    initSeries = pd.Series(ts[:p*sp])

    if mtype == 'additive':
        rawSeason = initSeries - \
            initSeries.rolling(min_periods=p, window=p, center=True).mean()
        initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
        initSeason = pd.Series(initSeason) - np.mean(initSeason)
        deSeasoned = [initSeries[v] - initSeason[v % p]
                      for v in range(len(initSeries))]
    else:
        rawSeason = initSeries / \
            initSeries.rolling(min_periods=p, window=p, center=True).mean()
        initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
        initSeason = pd.Series(initSeason) / \
            math.pow(np.prod(np.array(initSeason)), 1/p)
        deSeasoned = [initSeries[v] / initSeason[v % p]
                      for v in range(len(initSeries))]

    lm = linear_model.LinearRegression()
    lm.fit(pd.DataFrame(
        {'time': [t+1 for t in range(len(initSeries))]}), pd.Series(deSeasoned))
    return float(lm.intercept_), float(lm.coef_), list(initSeason)


def _MSD(tuning, *args):
    '''BFGS optimization to determine the optimal (alpha, beta, gamma) values'''

    predicted = []
    mtype = args[0]
    ts, p = args[1:3]
    Lt1, Tt1 = args[3:5]
    St1 = args[5][:]
    alpha, beta, gamma = tuning[:]

    for t in range(len(ts)):

        if mtype == 'additive':
            Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta * (Lt - Lt1) + (1 - beta) * (Tt1)
            St = gamma * (ts[t] - Lt) + (1 - gamma) * (St1[t % p])
            predicted.append(Lt1 + Tt1 + St1[t % p])
        else:
            Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta * (Lt - Lt1) + (1 - beta) * (Tt1)
            St = gamma * (ts[t] / Lt) + (1 - gamma) * (St1[t % p])
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
            Tt = beta * (Lt - Lt1) + (1 - beta) * (Tt1)
            St = gamma * (ts[t] - Lt) + (1 - gamma) * (St1[t % p])
            smoothed.append(Lt1 + Tt1 + St1[t % p])
        else:
            Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta * (Lt - Lt1) + (1 - beta) * (Tt1)
            St = gamma * (ts[t] / Lt) + (1 - gamma) * (St1[t % p])
            smoothed.append((Lt1 + Tt1) * St1[t % p])

        Lt1, Tt1, St1[t % p] = Lt, Tt, St

    MSD = sum([(ts[t] - smoothed[t]) **
               2 for t in range(len(smoothed))])/len(smoothed)
    return MSD, (Lt1, Tt1, St1), smoothed


def _predictValues(mtype, p, ahead, params):
    '''generate predicted values ahead periods into the future'''

    Lt, Tt, St = params
    if mtype == 'additive':
        return [Lt + (t+1)*Tt + St[t % p] for t in range(ahead)]
    else:
        return [(Lt + (t+1)*Tt) * St[t % p] for t in range(ahead)]

# Arima


def arima(tsA):
    # split into train and test sets
    size = int(len(tsA) * 0.66)
    train, test = tsA[0:size], tsA[size:len(tsA)]
    history = [x for x in train]
    predictions = list()
    result = []

    for t in range(len(train)):
        result.append(
            {'periode': t + 1, 'demande': train[t], 'prediction': train[t]})

    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))
        result.append(
            {'periode': size + t + 1, 'demande': obs, 'prediction': yhat})
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))

    return {'rmse': rmse, 'prevision': result}
