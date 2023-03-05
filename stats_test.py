import pandas as pd
import numpy as np
import os
import statsmodels as sm

Datasets = os.path.dirname(os.getcwd()) + "\\Data\\"
__depends__ = [Datasets+"RetailMarketOrder.sas7bdat", 
               Datasets+"InstitutionOrder.sas7bdat", 
               Datasets+"FutureReturn.sas7bdat"]

# load datasets, retailer logs and returns
retail = pd.read_sas(__depends__[0], encoding = 'latin-1')
retail.columns = retail.columns.str.lower()
retail[['permno','russellgroup']] = retail[['permno','russellgroup']].astype(int)
retail.tail()

institution = pd.read_sas(__depends__[1], encoding = 'latin-1')
institution.columns = institution.columns.str.lower()
institution[['permno','russellgroup']] = institution[['permno','russellgroup']].astype(int)
institution.tail()

ret = pd.read_sas(__depends__[2], encoding = 'latin-1')
ret.columns = ret.columns.str.lower()
ret['permno'] = ret['permno'].astype(int)
ret.tail()

# Cap size with retailers
ptflist = ['Low','2','3','4','5','6','7','8','9','High','H-L']
stock_grp = ['Large-Cap','Mid-Cap','Small-Cap','Micro-Cap','Nano-Cap']
results = pd.DataFrame([], columns = stock_grp, 
                       index = ['Low','','2','','3','','4','','5','','6','','7','','8','','9','','High','','H-L',''])
results.index.name = 'Retail Market Order Imbalance'

fut_ret = 'ret_5'
for i in np.arange(5):
    stock_grp[i]
    df_ = pd.merge(retail[retail['russellgroup'] == (i+1)][['permno','date','moribvol']], 
                   ret, on = ['permno','date'], how = 'inner')
    breakpoints = df_.groupby('date')['moribvol'].describe(percentiles = [.1,.2,.3,.4,.5,.6,.7,.8,.9])
    df_ = pd.merge(df_, breakpoints, on = 'date', how = 'left')
     
    def group(row, signal = 'moribvol'):
        if row[signal] < row['10%']:
            value = 'Low'
        elif row[signal] < row['20%']:
            value = '2'
        elif row[signal] < row['30%']:
            value = '3'
        elif row[signal] < row['40%']:
            value = '4'
        elif row[signal] < row['50%']:
            value = '5'
        elif row[signal] < row['60%']:
            value = '6'
        elif row[signal] < row['70%']:
            value = '7'
        elif row[signal] < row['80%']:
            value = '8'
        elif row[signal] < row['90%']:
            value = '9'
        elif row[signal] >= row['90%']:
            value = 'High'
        else:
            value = np.nan
        return value
        
    df_['rank'] = df_.apply(group, axis = 1) 
    ptf = df_.groupby(['date','rank'])[fut_ret].mean().unstack().dropna()
    ptf['H-L'] = ptf['High']-ptf['Low']
    ptf['const'] = 1
    for j in np.arange(11):
        res = sm.OLS(ptf[ptflist[j]], ptf['const'], missing = 'drop').fit(cov_type = 'HAC', cov_kwds = {'maxlags':1})
        results.iloc[2*j,i] = 252*res.params[0]
        results.iloc[2*j+1,i] = res.tvalues[0]
    
results

# Sector 
ptflist = ['Low','2','3','4','5','6','7','8','9','High','H-L']
stock_grp = ['XLU','XLF','XLB','XLP','XLY','XLK','XLV','XLI','XLE','XLC','XLR']
results = pd.DataFrame([], columns = stock_grp, 
                       index = ['Low','','2','','3','','4','','5','','6','','7','','8','','9','','High','','H-L',''])
results.index.name = 'Retail Market Order Imbalance'

fut_ret = 'ret_5'
for i in np.arange(11):
    stock_grp[i]
    df_ = pd.merge(retail[retail['sector'] == stock_grp[i]][['permno','date','moribvol']], 
                   ret, on = ['permno','date'], how = 'inner')
    breakpoints = df_.groupby('date')['moribvol'].describe(percentiles = [.1,.2,.3,.4,.5,.6,.7,.8,.9])
    df_ = pd.merge(df_, breakpoints, on = 'date', how = 'left')
     
    def group(row, signal = 'moribvol'):
        if row[signal] < row['10%']:
            value = 'Low'
        elif row[signal] < row['20%']:
            value = '2'
        elif row[signal] < row['30%']:
            value = '3'
        elif row[signal] < row['40%']:
            value = '4'
        elif row[signal] < row['50%']:
            value = '5'
        elif row[signal] < row['60%']:
            value = '6'
        elif row[signal] < row['70%']:
            value = '7'
        elif row[signal] < row['80%']:
            value = '8'
        elif row[signal] < row['90%']:
            value = '9'
        elif row[signal] >= row['90%']:
            value = 'High'
        else:
            value = np.nan
        return value
        
    df_['rank'] = df_.apply(group, axis = 1) 
    ptf = df_.groupby(['date','rank'])[fut_ret].mean().unstack().dropna()
    ptf['H-L'] = ptf['High']-ptf['Low']
    ptf['const'] = 1
    for j in np.arange(11):
        res = sm.OLS(ptf[ptflist[j]], ptf['const'], missing = 'drop').fit(cov_type = 'HAC', cov_kwds = {'maxlags':1})
        results.iloc[2*j,i] = 252*res.params[0]
        results.iloc[2*j+1,i] = res.tvalues[0]
    
results

# Cap size with institutions
ptflist = ['Low','2','3','4','5','6','7','8','9','High','H-L']
stock_grp = ['Large-Cap','Mid-Cap','Small-Cap']
results = pd.DataFrame([], columns = stock_grp, 
                       index = ['Low','','2','','3','','4','','5','','6','','7','','8','','9','','High','','H-L',''])
results.index.name = 'Institution Order Imbalance'

fut_ret = 'ret_1'
for i in np.arange(3):
    stock_grp[i]
    df_ = pd.merge(institution[institution['russellgroup'] == (i+1)][['permno','date','iibvol']], 
                   ret, on = ['permno','date'], how = 'inner')
    breakpoints = df_.groupby('date')['iibvol'].describe(percentiles = [.1,.2,.3,.4,.5,.6,.7,.8,.9])
    df_ = pd.merge(df_, breakpoints, on = 'date', how = 'left')
     
    def group(row, signal = 'iibvol'):
        if row[signal] < row['10%']:
            value = 'Low'
        elif row[signal] < row['20%']:
            value = '2'
        elif row[signal] < row['30%']:
            value = '3'
        elif row[signal] < row['40%']:
            value = '4'
        elif row[signal] < row['50%']:
            value = '5'
        elif row[signal] < row['60%']:
            value = '6'
        elif row[signal] < row['70%']:
            value = '7'
        elif row[signal] < row['80%']:
            value = '8'
        elif row[signal] < row['90%']:
            value = '9'
        elif row[signal] >= row['90%']:
            value = 'High'
        else:
            value = np.nan
        return value
        
    df_['rank'] = df_.apply(group, axis = 1) 
    ptf = df_.groupby(['date','rank'])[fut_ret].mean().unstack().dropna()
    ptf['H-L'] = ptf['High']-ptf['Low']
    ptf['const'] = 1
    for j in np.arange(11):
        res = sm.OLS(ptf[ptflist[j]], ptf['const'], missing = 'drop').fit(cov_type = 'HAC', cov_kwds = {'maxlags':1})
        results.iloc[2*j,i] = 252*res.params[0]
        results.iloc[2*j+1,i] = res.tvalues[0]
    
results