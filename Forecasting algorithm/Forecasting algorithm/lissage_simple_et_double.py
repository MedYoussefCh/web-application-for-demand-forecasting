#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt


def single_exponential_smoothing(series, alpha):
    result = [series[0]] 
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    result.append(alpha * series[n] + (1 - alpha) * result[n])
    plt.plot(series, color='g', label='series')
    plt.plot(result, color='r', label='result')
    plt.title('prevision')
    plt.xlabel('period')
    plt.ylabel('demand')
    plt.legend(loc='best')
    plt.show()
    return result

    
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
        
    plt.plot(series, color='g', label='series')
    plt.plot(result, color='r', label='result')
    plt.title('prevision')
    plt.xlabel('period')
    plt.ylabel('demand')
    plt.legend(loc='best')
    plt.show()
    return result


series = [3,10,12,13,12,10,12]

alpha = 0.9
beta = 0.9

print("Original series (order %d)" % len(series))
print(series)
print("----------------------")
ses = single_exponential_smoothing(series, alpha)
print("Single Exponential Smoothing (order %d)" % len(ses))
print(ses)
print("----------------------")
des = double_exponential_smoothing(series, alpha, beta)
print("Double Exponential Smoothing (order %d)" % len(des))
print(des)


# In[ ]:




