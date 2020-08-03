#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 01:13:06 2020

@author: averysmith

Stock Predictor 




List of Good Resources:
    - https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/
    - https://medium.com/auquan/https-medium-com-auquan-machine-learning-techniques-trading-b7120cee4f05
    - https://alphascientist.com/feature_engineering.html
    - https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7
    - https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/



"""

## Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np 
import yfinance as yf
from datetime import datetime
import fastai.tabular 
from ta import add_all_ta_features
from sklearn.metrics import mean_squared_error
import os
import glob


plt.ioff()

w = 16
h = 4

results = pd.DataFrame(columns = ['Stock','Day','Model','R2','MSE'])

tickers = ['AMZN','XOM','TSLA','DOMO','SNAP','UBER','ZM']


for i in tickers:
    ## Load in data 
    print(i)
    print('---------------------------------------')

    #define the ticker symbol
    tickerSymbol = i
    filename = 'data_' + tickerSymbol +'.csv'
    
    #get data on this ticker
       # Check if stored first
    extension = 'csv'
    csvs = glob.glob('*.{}'.format(extension))
        
    if filename in csvs:
           df = pd.read_csv(filename) 
    else:
        tickerData = yf.Ticker(tickerSymbol)
    
        #get the historical prices for this ticker
        df = tickerData.history(period='60m', start='2015-4-1', end='2020-7-19')
        
        df.to_csv('data_' + tickerSymbol +'.csv')
    
    
    date_change = '%Y-%m-%d'
    
    df['Date'] = pd.to_datetime(df['Date'], format = date_change)
    dates = df['Date']

    
    
    # Feature engineering from package
    df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)
    
    
    ## Plot the data 
    #plt.plot(df['Date'],df['High'])
    plt.figure(figsize=(w,h))
    plt.plot(df['Date'],df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close ($)')
    plt.tight_layout()
    plt.title(tickerSymbol)
    plt.tight_layout()
    plt.savefig(tickerSymbol+'.png')
    #plt.show()
    
    
    
    
    ## Adding more girth to the data      
        
    # adding date components
    # Regression
    fastai.tabular.add_datepart(df,'Date', drop = 'True')
    df['Date'] = pd.to_datetime(df.index.values, format = date_change)
    fastai.tabular.add_cyclic_datepart(df, 'Date', drop = 'True')
    
    
        
        
    
    ## Adding algorithms
    
    shifts = [1,5,10]
    df.insert(0,'Dates',dates)
    for j in shifts:    
        
        # add lag
        shiftdays = j
        shift = -shiftdays
        df['Close_lag'] = df['Close'].shift(shift)
        
        
        # change some columns to cat
        for col in df.columns[80:-10]:
            df[col] = df[col].astype('category')
            
        for col in df.columns[1:80]:
            df[col] = df[col].astype('float')
            
        for col in df.columns[-10:]:
            df[col] = df[col].astype('float')
        
            
        # Split into train and test
        train_pct = .75
        
        train_pt = int(len(df)*train_pct)
        if train_pt < 400:
            train_pt = train_pt = int(len(df)*.5)
        
        train = df.iloc[:train_pt,:]
        test = df.iloc[train_pt:,:]
        
        x_train = train.iloc[:shift,1:-1]
        y_train = train['Close_lag'][:shift]
        x_test = test.iloc[:shift,1:-1]
        y_test = test['Close'][:shift]
        
        
    
        
        
        
        # Linear regression
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train,y_train)
        lr_pred = lr.predict(x_test)
        lr_MSE = mean_squared_error(y_test, lr_pred)
        lr_R2 = lr.score(x_test, y_test)
        print('Linear Regression R2: {}'.format(lr_R2))
        print('Linear Regression MSE: {}'.format(lr_MSE))
        
        
        
        # plot
        plt.figure(figsize=(w,h))
        plt.plot(train['Dates'],train['Close'], label = 'True Training Data')
        plt.plot(test['Dates'],test['Close'], label = "True Test Data")
        plt.plot(test['Dates'][:shift], lr_pred, label = "prediction")
        plt.legend()
        plt.ylabel('Close ($)')
        plt.title('Linear Regression - ' + tickerSymbol + ' - ' + str(shiftdays))
        plt.tight_layout()
        plt.savefig('Linear Regression - ' + tickerSymbol + ' - ' + str(shiftdays) + '.png')
        #plt.show()
        
        row_dict = {'Stock' : tickerSymbol,'Day': shiftdays,'Model': 'Linear Regression','R2':lr_R2 ,'MSE':lr_MSE}
        results_temp = pd.DataFrame(row_dict, index = [0])
        results = results.append(results_temp, ignore_index=True)
        
        
        
        
        # Linear regression in OLS
        import statsmodels.api as sm
        
        the_cats = df.columns[83:-10]
        a = pd.get_dummies(df[the_cats])
        df_ols = pd.concat([df,a],axis=1)
        df_ols = df_ols.drop(the_cats,axis=1)
        df_ols = df_ols.drop(['Dates','Close_lag'], axis =1)
        
        train_ols = df_ols.iloc[:train_pt,1:]
        test_ols = df_ols.iloc[train_pt:,1:]
        
        x_train_ols = train_ols.iloc[:shift,:-1]
        x_test_ols = test_ols.iloc[:shift,:-1]
        
        
        
        
        ols = sm.OLS(y_train,np.asmatrix(x_train_ols, dtype=float))
        ols_results = ols.fit()
        ols_results.summary()
        
        f = ols_results.pvalues
        #f = f.drop('const',axis=0)
        f = pd.DataFrame(f)
        f['name'] = x_train_ols.columns
        f = f.reset_index()
        f = f.dropna()
        
        # importance graph
        top_n = 10
        top_n_df = np.argsort(f.iloc[:,1])[:top_n]
        top_n_idx = top_n_df.values
        top_n_indicators = x_train_ols.columns[top_n_idx]
        top_n_pvalues = [f.iloc[i,1] for i in top_n_idx]
        
        x_pos = [i for i, _ in enumerate(top_n_indicators)]
        
        
        plt.figure(figsize=(w,h))
        plt.plot(top_n_pvalues, top_n_indicators, color='green')
        plt.xticks(rotation=90)
        plt.title('Sig P-values Linear Regression - ' + tickerSymbol + ' - ' + str(shiftdays))
        plt.tight_layout()
        plt.savefig('LinearRegression P Values _ ' + tickerSymbol + ' - ' + str(shiftdays) + '.png')
        #plt.show()
        

        
        
        
        
        # Random forest 
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=1000,max_depth=10)
        rf.fit(x_train,y_train)
        rf_pred = rf.predict(x_test)
        rf_MSE = mean_squared_error(y_test, rf_pred)
        rf_R2_train = rf.score(x_train, y_train)
        rf_R2_test = rf.score(x_test, y_test)
        print('Random Forest R2 Test: {}'.format(rf_R2_train))
        print('Random Forest R2 Test: {}'.format(rf_R2_test))
        print('Random Forest MSE: {}'.format(rf_MSE))
        
        # plot
        plt.figure(figsize=(w,h))
        plt.plot(train['Dates'],train['Close'], label = 'True Training Data')
        plt.plot(test['Dates'], test['Close'], label = "True Test Data")
        plt.plot(test['Dates'][:shift], rf_pred, label = "prediction")
        plt.legend()
        plt.ylabel('Close ($)')
        plt.title('RandomForest - ' + tickerSymbol + ' - ' + str(shiftdays))
        plt.tight_layout()
        plt.savefig('RandomForest - ' + tickerSymbol + ' - ' + str(shiftdays) + '.png')
        #plt.show()
        
        
        row_dict = {'Stock' : tickerSymbol,'Day': shiftdays,'Model': 'Random Forest','R2':rf_R2_test ,'MSE':rf_MSE}
        results_temp = pd.DataFrame(row_dict, index = [0])
        results = results.append(results_temp, ignore_index=True)
        
        
        
        
        # Neural nets 
        from sklearn.preprocessing import StandardScaler 
        
        # Scaling data
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        
        from sklearn.neural_network import MLPRegressor
        MLP = MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes = (100,), activation = 'identity',learning_rate = 'adaptive').fit(x_train_scaled, y_train)
        MLP_pred = MLP.predict(x_test_scaled)
        MLP_MSE = mean_squared_error(y_test, MLP_pred)
        MLP_R2 = MLP.score(x_test_scaled, y_test)
        
        print('Muli-layer Perceptron R2 Test: {}'.format(MLP_R2))
        print('Multi-layer Perceptron MSE: {}'.format(MLP_MSE))
        
        
        # plot
        plt.figure(figsize=(w,h))
        plt.plot(train['Dates'],train['Close'], label = 'True Training Data')
        plt.plot(test['Dates'],test['Close'], label = "True Test Data")
        plt.plot(test['Dates'][:shift], MLP_pred, label = "prediction")
        plt.legend()
        plt.ylabel('Close ($)')
        plt.title('Multi-layer Perceptron - ' + tickerSymbol + ' - ' + str(shiftdays))
        plt.tight_layout()
        plt.savefig('Multi-layer Perceptron - ' + tickerSymbol + ' - ' + str(shiftdays) + '.png')
        #plt.show()
        
        
        row_dict = {'Stock' : tickerSymbol,'Day': shiftdays,'Model': 'MLP','R2':MLP_R2 ,'MSE':MLP_MSE}
        results_temp = pd.DataFrame(row_dict, index = [0])
        results = results = results.append(results_temp, ignore_index=True)
        
        
        

 
        plt.close('all')
    


plt.close('all')
results.to_excel('StockResults.xlsx')


















