# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:47:44 2021

@author: jared
"""

import numpy as np
import pandas as pd
import statsmodels.tsa.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import datetime
warnings.filterwarnings("ignore")

def tsplot(y, lags=50, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))    
        y.plot(ax=ts_ax)
        p_value = sm.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        sm.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        sm.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        plt.tight_layout()
    return

def write_results(file_name, results, columns):
    with open(file_name,"w") as fw:
        print(results.summary(), file=fw)
        fw.write("================test_whiteness================\n")
        print(results.test_whiteness(nlags=20), file=fw)
        fw.write("================results.roots()================\n")
        for root in results.roots:
            fw.write(str(root)+",")
        fw.write("\n================is_stable================\n")
        fw.write(str(results.is_stable()))
        fw.write("\n================granger causality================\n")
        for V1 in columns:
            for V2 in columns:
                fw.write("Granger test: "+V2+"-->"+V1+"\n")
                print(results.test_causality(V1, [V2],kind='f').summary(), file=fw)
                fw.write("\n")

        fw.write("\n================long_run effects================\n")
        for effect in results.long_run_effects():
            fw.write(str(effect)+",")
        fw.write("\n================FEVD values================\n")
        from contextlib import redirect_stdout
        with redirect_stdout(fw):
            print(results.fevd(20).summary())

def structural_analyses(file_name, results, lag=20):
    # impulse response analyses
    irf = results.irf(lag)
    irf.plot(orth=False)
    plt.savefig(file_name+"_irf_noorth.png")
    # irf.plot(impulse='V1', response="V2")
    irf.plot_cum_effects(orth=False)
    plt.savefig(file_name+"_irf_cum_noorth.png")
    results.long_run_effects()

    # forecast error decomposition
    results.fevd(lag).plot()
    plt.savefig(file_name+"_fevd.png")
    
    
# read csv file    
data = pd.read_csv("C:\\NUS\\Y2S2\\BT4014\\Final Project\\updated_crypto_data.csv")
# convert to date format
data['date'] = pd.to_datetime(data['date'], format = "%Y-%m-%d").dt.date
# set date as index
data.index = data['date']
data.drop(['Unnamed: 0', 'date'], axis=1, inplace=True)

#remove first 6 rows so that data can be aggregated to weekly
data = data.iloc[6:]
data.index = pd.to_datetime(data.index)
agg_data = data.resample('7D').mean() # mean
print(agg_data)



# Tether

id = 'tether'
data_tether = agg_data.drop(['XLM', 'ETH','NANO'], axis=1, inplace = False)
print(data_tether)

focused_columns = ["USD", "BTC", "USDT"]
# see original plots of all products
for column in focused_columns:
    tsplot(data_tether[column], lags = 30)
    plt.savefig("C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_tsplot_'+column+'_weekly.png')
    plt.show()
    plt.close()

# log transformation
for column in focused_columns:
    data_tether[column] = np.log(data_tether[column]) 
    tsplot(data_tether[column], lags = 30)
    plt.savefig("C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_tsplot_log_'+column+'_weekly.png')
    plt.show()
    plt.close()

# first order differencing
data_tether_diff = data_tether.diff().dropna()
for column in focused_columns:
    tsplot(data_tether_diff[column], lags = 30)
    plt.savefig("C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_tsplot_diff_'+column+'_weekly.png')
    plt.show()
    plt.close()

# modelling using VAR for analysis
tether_model = VAR(data_tether_diff.dropna())
print(tether_model.select_order(20).summary())
selected_orders = tether_model.select_order(20).selected_orders
print(selected_orders)

selected_orders_vals = list(selected_orders.values())
bic_selected_order = selected_orders_vals[1]
print(bic_selected_order)

# order 1 is the best lag order
results = tether_model.fit(bic_selected_order, trend="c")
results_file = "C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_results_lag'+str(bic_selected_order)+'_weekly.txt'
structural_file = "C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_lag'+str(bic_selected_order)+ '_weekly'
columns = data_tether.columns.values.tolist()
write_results(results_file, results, columns)
structural_analyses(structural_file, results, 20)



# Ethereum
id = 'ethereum'
data_eth = agg_data.drop(['XLM', 'USDT','NANO'], axis=1, inplace = False)
print(data_eth)

focused_columns = ["USD", "BTC", "ETH"]
# see original plots of all products
for column in focused_columns:
    tsplot(data_eth[column], lags = 30)
    plt.savefig("C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_tsplot_'+column+'_weekly.png')
    plt.show()
    plt.close()

# log transformation
for column in focused_columns:
    data_eth[column] = np.log(data_eth[column]) 
    tsplot(data_eth[column], lags = 30)
    plt.savefig("C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_tsplot_log_'+column+'_weekly.png')
    plt.show()
    plt.close()

# first order differencing
data_eth_diff = data_eth.diff().dropna()
for column in focused_columns:
    tsplot(data_eth_diff[column], lags = 30)
    plt.savefig("C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_tsplot_diff_'+column+'_weekly.png')
    plt.show()
    plt.close()

# modelling using VAR
eth_model = VAR(data_eth_diff.dropna()) 
print(eth_model.select_order(20).summary())
selected_orders = eth_model.select_order(20).selected_orders
print(selected_orders)

selected_orders_vals = list(selected_orders.values())
bic_selected_order = selected_orders_vals[1]
print(bic_selected_order)

# order 1 is the best lag order
results = eth_model.fit(bic_selected_order,trend="c")
results_file = "C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_results_lag'+str(bic_selected_order)+'_weekly.txt'
structural_file = "C:\\NUS\\Y2S2\\BT4014\\Final Project\\" + str(id)+'_lag'+str(bic_selected_order) + '_weekly'
columns = data_eth.columns.values.tolist()
write_results(results_file, results, columns)
structural_analyses(structural_file, results, 20)




