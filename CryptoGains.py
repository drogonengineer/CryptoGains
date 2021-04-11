#https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1618099200&end=1618160400&period=900
#https://www.epochconverter.com/
#https://docs.poloniex.com/#returnchartdata
import json
import matplotlib.pyplot as plt
import numpy as np


def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """

    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema


ip_file_name = 'eth_1_4_to_11_4_1800seconds.json'

# Opening JSON file
f = open(ip_file_name,)

# returns JSON object as 
# a dictionary
data = json.load(f)

open_data = []
# Iterating through the json
# list
for i in data:
    #print(type(i))
    #print(i)
    #print(i["open"])
    open_price = i["open"]
    open_data.append(float(open_price))


ema_9 = ema(open_data, 9)
ema_20 = ema(open_data, 20)

ema_9 = np.array(ema_9)
ema_20 = np.array(ema_20)

len_ema_9 = len(ema_9)
len_ema_20 = len(ema_20)
x = np.arange(0, len_ema_20)

ema_9 = ema_9[0:len_ema_20]



plt.plot(x, ema_9, '-')
plt.ylabel('EMA every 9 and 20 points')
plt.plot(ema_20)

idx = np.argwhere(np.diff(np.sign(ema_9 - ema_20))).flatten()
plt.plot(x[idx], ema_9[idx], 'ro')

print(ema_9[idx])
print(ema_20[idx])
potential_profit_list = (ema_20[idx])


potential_profit = [x - potential_profit_list[i - 1] for i, x in enumerate(potential_profit_list)][1:]

print(potential_profit)



plt.show()

# Closing file
f.close()




