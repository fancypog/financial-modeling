# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:53:37 2023

@author: FK
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:05:03 2023

@author: FancyPog Dev
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy . stats as scs
import yfinance as yf
from arch import arch_model
import arch
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.stats import ttest_1samp
from arch.univariate import ARCHInMean, ARX, GARCH
from datetime import datetime, timedelta


# =============================================================================
# TASK ONE
# =============================================================================

# =============================================================================
# 1.1
# =============================================================================

# Set working console directory and import the file
df = pd.read_csv('GOOG_202001.csv.gz', compression = 'gzip')
total = len(df.index)

# F1
L1 = ['SIZE', 'PRICE', 'NBO', 'NBB', 'NBOqty', 'NBBqty']
[df.drop(df[df[i] <= 0].index, inplace = True) for i in L1]
F1 = total - len(df.index)

# F2
df = df.drop(df[df['NBO'] - df['NBB'] <= 0].index)
F2 = total - len(df.index) - F1

# F3
df = df.groupby(['DATE', 'TIME_M', 'EX', 'BuySell']).agg({'PRICE': 'median', \
                    'NBO': 'median','NBB': 'median','SIZE':'sum', 'NBOqty':'sum', 
                    'NBBqty': 'sum'}).reset_index()
F3 = total - len(df.index) -F2

# F4
# Calculate spread and daily median spread, store the latter seperately for later reference
df['spread'] = df['NBO'] - df['NBB']
medspread = df.groupby('DATE')['spread'].median()

# Calculate the 50 times spread difference and store it as a new column
# Here we use matching 'DATE' index to retrieve corresponding median spread from medspread Dataframe
df['spreaddiff'] = df.apply(lambda row: row['spread'] - 50 * medspread[row['DATE']], axis=1)

# Drop items with a positve 50 times spread difference
df = df.drop(df[df['spreaddiff'] > 0].index)
F4 = total - len(df.index) - F3

# F5
df = df.drop(df[df['PRICE'] > (df['NBO'] + df['spread'])].index)
df = df.drop(df[df['PRICE'] < (df['NBB'] - df['spread'])].index)
F5 = total - len(df.index) - F4

# Summarise
number = [{'F1': F1, 'F2': F2,'F3': F3, 'F4': F4, 'F5': F5}]
summary = pd.DataFrame(number, index=['Number'])
summary.loc['Proportion'] = 100* summary.loc['Number']/total
   
# =============================================================================
# 1.2
# =============================================================================

# create a timelapse variable and set it as index for furthur calculations
df['Stamp'] = pd.to_datetime( df['DATE']) + df['TIME_M'].apply(lambda x: pd.Timedelta (x , unit ='sec'))
df.set_index('Stamp', drop = True, inplace = True)

# Compute returns for furthur use
df['Return'] = np.log( df['PRICE']/ df['PRICE'].shift(1))
ret = df['Return'].dropna()

# Create a list of frequencies for later iteration
freqs = ['1s', '2s', '3s', '4s', '5s', '10s', '20s', '30s', '40s', '50s', '1min',\
         '2min','3min','4min', '5min', '6min', '7min', '8min', '9min', \
             '10min', '15min', '20min', '30min']
  
# Define functions to compute and store RV, BV and TRV
def computeRV(ret, freqs):
    # Create a dictionary to store results in the process
    RV_dict = {}
    ret. dropna(inplace = True)
    # group returns by date for later calculations for each day
    grouped = ret.groupby(ret.index.date)
    # compute RV for each trading day
    for freq in freqs:
        RV = grouped.apply(lambda x: sum(x.resample(rule=freq, closed='right', label='right').apply('sum') ** 2))
        # store the results
        RV_dict[freq] = RV
    # convert all results to a dataframe
    RV_df = pd.DataFrame(RV_dict)
   
    return RV_df

def computeBV(ret, freqs):
    # Create a dictionary to store results in the process
    BV_dict = {}
    ret. dropna(inplace = True)
    # group returns by date for later calculations for each day
    grouped = ret.groupby(ret.index.date)
    # Compute BV
    for freq in freqs:
        ret_freq = ret.resample ( rule = freq, closed ='right', label ='right').apply ('sum')
        ret2 = np.absolute(ret_freq).values
        BV =  grouped.apply(lambda x: np.pi/2*sum(ret2[0:-1]*ret2[1:]))
        BV_dict[freq] = BV
    # convert all results to a dataframe
    BV_df = pd.DataFrame(BV_dict)
    
    return BV_df 

def computeTRV(ret, freqs):
    # Call the two functions above to get RV and BV dataframes
    RVdf = computeRV(ret, freqs)
    BVdf = computeBV(ret, freqs) 
    # Compute TRV
    min_RV_BV = RVdf.combine(BVdf, np.minimum)
    u = 3/np.sqrt(len(RVdf))*np.sqrt(min_RV_BV)
    TRV = (RVdf <= u) * RVdf + (RVdf > u) * (BVdf + 2 * RVdf - 2 * u)
    return TRV

# Compute and store RV, BV and TRV
RVdf = computeRV(ret, freqs)
BVdf = computeBV(ret, freqs) 
TRVdf = computeTRV(ret, freqs)


# =============================================================================
# 1.3
# =============================================================================

# Scale the data
RV_scale = 10000* RVdf
BV_scale = 10000* BVdf
TRV_scale = 10000* TRVdf

# Define a function to plot one volatility signiture plot
def plotVolatility(ax, Dataframe, measure):
    # Compute Average RVs/ BVs/ TRVs across different days
    average = Dataframe. mean()
    time =[0] + [ pd. Timedelta(x). total_seconds() for x in average. index[1:]]
    # Plot
    sns. scatterplot(ax=ax, x= time, y = average)
    ax. set_xlabel ('Sampling frequency (secs)', fontsize =12)
    ax. set_ylabel ('Averaged ' + measure +' (x$10^{-4}$)', fontsize =12)
    
# Make the plots using the function above
# Put them in a convas and set the titles
fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(16, 4))
plt.subplots_adjust(hspace=0.3) # set the wide between the gap
plotVolatility(axes[0], RV_scale, 'RV')
axes[0].set_title('RV signature plot', fontsize =12)
plotVolatility(axes[1], BV_scale, 'BV')
axes[1].set_title('BV signature plot', fontsize =12)
plotVolatility(axes[2], TRV_scale, 'TRV')
axes[2].set_title('TRV signature plot', fontsize =12)
plt.show()


# =============================================================================
# 1.4
# =============================================================================

# Define a jump test function
def jumpTest(RV, BV, sigLevel):
    # Create a Dataframe to store results in the process
    jumpdf = pd.DataFrame()
    # Slice the sampling frequecy data fro RV and BV Dataframes
    jumpdf['RV'] = RV
    jumpdf['BV'] = BV
    # Calculate and store Jt
    Jt = (RV - BV). apply(lambda x: max(x, 0))
    jumpdf['J'] = Jt
    # Calculate test statistics
    dt = 5/(60*6.5)
    mu = 2**(2/3) * math . gamma (7/6) / math . gamma (1/2)
    TP = 1/ dt * mu **( -3) *sum (( ret [0: -2]* ret [1: -1]* ret [2:]) **(4/3) )
    z = Jt / np . sqrt ( TP * dt *( np . pi **2/4+ np .pi -5) )
    pvalue = 1 - scs . norm . cdf (z)
    # Test and store boolean data of whether or not there are jumps
    jumpdf['jump'] = np.where(pvalue < sigLevel, 'Yes', 'No')
    return jumpdf

jumpdf = jumpTest(RVdf['5min'], BVdf['5min'], 0.05)


# =============================================================================
# TASK TWO
# =============================================================================

# Import the file
df2 = pd.read_csv('SP100-Feb2023.csv')

# Exclude the stocks
L1 = ['ABBV', 'AVGO', 'CHTR', 'DOW', 'GM', 'KHC', 'META', 'PYPL', 'TSLA']
[df2.drop(df2[df2.Ticker == i].index, inplace = True) for i in L1]

# reset a new index for the Dataframe in case we draw the index of stocks excluded
df2 = df2.reset_index(drop = True)

# Draw a random sample of 2 stocks
np.random.seed(36176428)
L2 = np.random.randint(0, df2.index[-1], 2).tolist()

# Get the stock list from
stock_list = [df2.iloc[i].Ticker for i in L2]
stock_list.sort()

# Download the data
data = yf.download(stock_list, start="2009-01-01", end="2022-12-31")['Adj Close']
log_ret = (np.log(data) - np.log(data.shift(1))).dropna()
log_ret = log_ret*100 # transter into percentage


# =============================================================================
# 2.1
# =============================================================================

# Define the enddate and in-sample log returns
enddate = '2018-12-31'
logr_1 = log_ret[log_ret.index <= enddate]

# Define ranges of parameters
m_range = range(0,4)
q_range = range(0,4)
p_range = range(1,4)

# Define a function to choose the best fit model based on minAIC
def find_and_print_BestFit(logr_1, stock_ticker, m_range, p_range, q_range):
    # Create a list to store aic values along the way 
    aic = []
    # Calculate 
    for p in p_range:
        for o in range(0, p+1): # This loop make sure o is no greater than p
            for q in q_range:
                for m in m_range:
                    gjr_garch_obj = arch.arch_model(logr_1[stock_ticker], p=p, o=o, q=q, dist='StudentsT')
                    gjr_garch_fit = gjr_garch_obj.fit(disp='off')
                    aic.append([m, p, o, q, gjr_garch_fit.aic])      
    # Convert the list of AIC values to a DataFrame, store the parameters and AIC for later print
    aic_df = pd.DataFrame(aic, columns=['m', 'p', 'o', 'q', 'AIC'])
    # Find the index of the row with minimal AIC
    min_aic_index = aic_df['AIC'].idxmin()
    # Store the parameters
    # use int() for m, p, o and q so the print looks pretty (They're in one decimal point in the Dataframe);
    m_val = int(aic_df.iloc[min_aic_index]['m'])
    p_val = int(aic_df.iloc[min_aic_index]['p'])
    o_val = int(aic_df.iloc[min_aic_index]['o'])
    q_val = int(aic_df.iloc[min_aic_index]['q'])
    AIC_val = aic_df.iloc[min_aic_index]['AIC']
    # Print the result using the index above to retrieve corresponding values
    # use str() so you can concatenate everything in one print()
    print('Best-fitted AR(m)-GJR-GARCH(p,o,q) model for ' + stock_ticker + ': AR(' + \
          str(m_val) + ')-GJR-GARCH(' + str(p_val) + ',' + str(o_val) + ',' + str(q_val) + \
                  ')' + ' - AIC = ' + str(AIC_val))
    # Store and return the parameters of the best fit model to a Series for later test and fitting
    data = {'Ticker': stock_ticker, 'm': m_val, 'p': p_val, 'o': o_val, 'q': q_val}
    best_params = pd.Series(data)
    return best_params

# Print the best fitted model by calling the function above 
best_fit_model_data1 = find_and_print_BestFit(logr_1, stock_list[0], m_range, p_range, q_range) #for the first stock CRM
best_fit_model_data2 = find_and_print_BestFit(logr_1, stock_list[1], m_range, p_range, q_range) #for the second stock WBA


# =============================================================================
# 2.2
# =============================================================================

# Define a function to test Leverage Effect
# ret is the Dataframe containing the returns
# model_data is the bestfitted_model_result data stored in 2.1, from which this function retrieve ticker, m, p, o, q
# so it makes calling this function less pain, quite automatic! hmm!
# sigLevel is the significance level for the test
def testLeverageEffect(ret, model_data, sigLevel):
    # Retrieve the data from the best-fitted-model data
    ticker = model_data.Ticker
    m = model_data.m
    p = model_data.p
    o = model_data.o
    q = model_data.q
    # Define and fit the model    
    arm_gjrgarch_poq = arch_model(100 * ret[ticker], mean='AR', lags=m, vol='GARCH', \
                                  p=p, o=o, q=q, power=1)
    arm_gjrgarch_poq_fit = arm_gjrgarch_poq.fit(update_freq=5)
    # Get the t-stat from the model result and calculate p-value of gamma
    t_stat_gamma = arm_gjrgarch_poq_fit.tvalues['gamma[1]']
    p_value = stats.norm.sf(abs(t_stat_gamma)) * 2 
    # Compare and print the results
    if p_value < sigLevel:
        print('For ' + ticker + ', leverage effect is present at the ' + str(sigLevel * 100) + '% significance level.' + '\n')
    else:
        print('For '+ ticker + ', leverage effect is not present at the ' + str(sigLevel * 100) + '% significance level.' + '\n')

# Test for leverage effect
testLeverageEffect(log_ret, best_fit_model_data1, 0.05) #for the first stock CRM
testLeverageEffect(log_ret, best_fit_model_data2, 0.05) #for the second stock WBA


# =============================================================================
# 2.3
# =============================================================================
# Define a function to fit the models
def fitGarchModel(ret, model_data):
    model = arch_model(ret[model_data.Ticker], mean = 'AR', lags = model_data.m, vol = 'Garch', \
                       p = model_data.p, o = model_data.o, q = model_data.q, dist='StudentsT')
    return model.fit()


# Define a function to make different plots, one plot per call, of course
# model_data is needed to retrieve the parameters so we can set titles automatically
def makePlot(ax, model_data, fitted_model_result, category):
    # Calculations of standard residuals
    stdresid = fitted_model_result.resid / fitted_model_result.conditional_volatility
    # Obtain fitted conditional volatility from the model
    cond_vol = fitted_model_result.conditional_volatility
    # This is used for setting titles of plots
    ticker = model_data.Ticker
    m = model_data.m
    p = model_data.p
    o = model_data.o
    q = model_data.q
    # 1) Time series of the standardised residuals
    if category in ['TS_stdresid', 1]:
        sns.lineplot(data = stdresid, ax=ax)
        ax.set_title('AR(' + str(m) + ')-GJR-GARCH(' + str(p) + ',' + str(o) + ',' + str(q) + \
                      ') ' + 'Standardized residuals-' + ticker)
        ax.tick_params(axis='x', labelrotation = 30) #rotate x label a bit so it doesn't look crowded
    # 2) Histogram of the standardised residuals
    elif category in ['His_stdresid', 2]:
       # Get the degree of freedom
        df = fitted_model_result.params['nu']
        # Calculate the range of values to plot
        x = np.linspace(stats.t.ppf(0.001, df), stats.t.ppf(0.999, df), 100)
        # Calculate the probability density function of the T-distribution
        t_pdf = stats.t.pdf(x, df)
        # Plot the histogram and T-distribution density function
        sns.histplot(stdresid, kde=True, stat='density', ax=ax)
        ax.plot(x, t_pdf, 'g', lw=2,  label=f't(df={df:.1f})')
        ax.set_title('Distribution of standardized residuals-' + ticker)
        ax.legend()
    # 3) ACF of the standardised residuals 
    elif category in ['ACF_stdresid', 3]:
        plot_acf(stdresid, ax= ax)
        ax.set_title('ACF of standardized residuals-' + ticker)
    # 4) ACF of the squared standardised residuals   
    elif category in ['ACF_sqrresiduals', 4]:
        plot_acf(stdresid **2, ax= ax)
        ax.set_title('ACF of standardized residuals squared-'+ ticker)
    # 5) Time series of the fitted conditional volatility 
    elif category in ['TS_convolatility', 5]:
        sns.lineplot(x = cond_vol.index, y = cond_vol, ax =ax)   
        ax.set_title('Fitted conditional volatility-' + ticker)
        ax.tick_params(axis='x', labelrotation = 30)
    else: print('category not recognised.')

# Fit the model first
model1_fit_result = fitGarchModel(logr_1, best_fit_model_data1)
model2_fit_result = fitGarchModel(logr_1, best_fit_model_data2) 
       
# Define a function to make the fist row of required plots using the function above
def makeRow(fitted_model_result, model_data, rowNumber):
    [makePlot(axes[rowNumber-1,i], model_data, fitted_model_result, i+1) for i in range(5)]

# Plot everything
# Set the size and number of slots of the canvas
fig, axes = plt.subplots(nrows=2,ncols=5,figsize=(30, 9)) 
plt.subplots_adjust(hspace=0.3) # set the wide between the gap    
makeRow(model1_fit_result, best_fit_model_data1, 1) # Plot the first row
makeRow(model2_fit_result, best_fit_model_data2, 2) # Plot the second row

# Draw conclusions and print the comments
print('From the plots above I conclude:' + '\n' + 
      '1. The standardized residuals of both stocks have a mean close to zero and are approximately normally distributed.' + '\n'
      '2. The autocorrelation function (ACF) of the standardized residuals drops off quickly and fluctuates around zero, indicating that there is little evidence of autocorrelation.' \
          + '\n' +
      '3. The fitted conditional volatility of both stocks varies over time, with CRM having a wider range of variation compared to WDA.'+ '\n')

# =============================================================================
# 2.4
# =============================================================================
# Define a function to conduct ARCH LM test
# fitted_model_result: the result of fitted model stored previously
def testARCHLM(model_data, fitted_model_result, sigLevel):
    
    # Derive model data for printing
    ticker = model_data.Ticker
    m = model_data.m
    p = model_data.p
    o = model_data.o
    q = model_data.q
    
    # Derive data from fitted model result for later calculation of t-stats
    residuals = fitted_model_result.resid
    max_lag = 10 

    # Compute the squared residuals
    squared_resid = np.square(residuals)
        
    # Build a DataFrame with the squared residuals and lagged squared residuals
    lagged_resid = [pd.Series(squared_resid).shift(i) for i in range(1, max_lag + 1)]
    lagged_resid = pd.concat(lagged_resid, axis=1)
    lagged_resid.columns = [f'squared_resid_lag{i}' for i in range(1, max_lag + 1)]
    df_lmtest = pd.concat([pd.Series(squared_resid), lagged_resid], axis=1)
    df_lmtest.columns = ['squared_resid'] + [f'squared_resid_lag{i}' for i in range(1, max_lag + 1)]
    df_lmtest.dropna(inplace=True)
    
    # Fit a linear regression model for the squared residuals
    X = df_lmtest.drop(columns=['squared_resid'])
    Y = df_lmtest['squared_resid']
    reg_model = sm.OLS(Y, sm.add_constant(X)).fit()
    
    # Compute the LM test statistic and the p-value
    R2 = reg_model.rsquared
    n = len(df_lmtest)
    LM = n * R2
    p_value = 1 - chi2.cdf(LM, max_lag)

    # Compare and print the results    
    if p_value < sigLevel:
        print('There are some remaining ARCH effects up to order 10 left in the standardized residuals of the AR(' \
              + str(m) + ')-GJR-GARCH(' + str(p) + ',' + str(o) + ',' + str(q) + ') model for stock ' + ticker + '.')
    else:
        print('There are no ARCH effects up to order 10 left in the standardized residuals of the AR(' + \
              str(m) + ')-GJR-GARCH(' + str(p) + ',' + str(o) + ',' + str(q) + ') model for stock ' + ticker + '.')

# Test for LM 
testARCHLM(best_fit_model_data1, model1_fit_result, 0.05)
testARCHLM(best_fit_model_data2, model2_fit_result, 0.05)

print('\n' + 'From the LM test above I conclude:' + '\n' + 
      'The volatility of CRM is not fully captured by the GARCH model, and that of WBA is well captured.' )


# =============================================================================
# 3.1
# =============================================================================
# From 2.1 we get the best fitted models(in-sample) are:
model1_fit_result
model2_fit_result

# Define function to do one-step forecast
# data: the whole sample data, including both in-sample and out-sample data
# start_date: the start of forecasting date
# stock: stock ticker
# n_model: name of the fitted model

def fcast_1step(ret, start_date, stock, fittedmodel):
    log_oos = ret.loc[start_date:].copy()
    n_output = log_oos.copy()[[]]
    for fdate in log_oos.index:
        y = ret.loc[ret.index < fdate, stock]
        n_model_ext = fittedmodel.forecast(y)
        fcast_mean = n_model_ext.mean.iloc[-1]
        fcast_var = n_model_ext.variance.iloc[-1]
        n_output.loc[fdate, ['f', 'fl', 'fu', 'volf']] = [fcast_mean, fcast_mean-1.96*np.sqrt(fcast_var),
                                                          fcast_mean+1.96*np.sqrt(fcast_var), np.sqrt(fcast_var)]
    return n_output

fcast1 = fcast_1step(log_ret, '2018-12-31', 'CRM', model1_fit_result)
fcast2 = fcast_1step(log_ret, '2018-12-31', 'WBA', model2_fit_result)

fcast1 = fcast_1step(log_ret, enddate, stock_list[0], model1_fit_result)
fcast2 = fcast_1step(log_ret, enddate, stock_list[1], model2_fit_result)






