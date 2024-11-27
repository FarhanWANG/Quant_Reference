# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:23:28 2024

@author: hazc
"""

import os
import numpy as np
import pandas as pd
from factor_cal_utils import standardize
import warnings
warnings.filterwarnings("ignore")

fdir = r'E:\跳跃因子'

def get_index(date, dates):
    dates = list(pd.to_datetime(dates))
    date = pd.to_datetime(date)
    try:
        idx = dates.index(date)
        return idx 
    except ValueError:
        dates.append(date)
        dates.sort()
        idx = dates.index(date) - 1
        return idx 

def open_mintable(fname, datecol='datetime', columns=None, stime=None, etime=None,
                  callback=None, **kwargs):
    fpath = os.path.join(fdir, 'raw_data', 'mindata', fname)
    qres = pd.read_parquet(fpath, columns=columns)
    if stime and etime:
        date_indexer = qres.index.get_level_values(datecol).indexer_between_time(stime, etime)   
        qres = qres.iloc[date_indexer]

    if callable(callback):
        qres = callback(qres, **kwargs)
    return qres

def jump(df):
    close = df['close'].unstack()
    single_ret = close / close.shift(1) - 1
    compound_ret = np.log(close / close.shift(1))
    delta = single_ret - compound_ret
    tailor_resid = delta * 2 - compound_ret ** 2
    jump = tailor_resid.mean(axis=0)
    
    jump = jump.to_frame('jump')
    jump = jump.assign(tradedate=df.index.levels[0].date[0])
    return jump

def amend_swing(high, low, close, jump_d, rwindow=20): 
    swing = (high - low) / close.shift(1)
    swing_1 = swing.where(jump_d.ge(jump_d.mean(axis=1), axis=0), -swing)

    single_ret = high / low.shift(1) - 1
    compound_ret = np.log(high / low.shift(1))
    delta = single_ret - compound_ret
    tailor_resid = delta * 2 - compound_ret ** 2
    swing_2 = swing.where(tailor_resid.ge(tailor_resid.mean(axis=1), axis=0), -swing)
    
    swing_1 = swing_1.rolling(rwindow).mean()
    swing_2 = swing_2.rolling(rwindow).mean()
    amend_swing = swing_1 + swing_2
    return amend_swing        

def jump_m(dailydf, window=20): #jump_d
    jump_mean = dailydf.rolling(window).mean()
    jump_std = dailydf.rolling(window).std()
    jump_m = jump_mean + jump_std
    return jump_m

def moth_to_fire(jump_d, high, low, close, window=20): #jump_d
    jump = jump_m(jump_d, window) #or use jumd_d directly
    a_swing = amend_swing(high, low, close, jump_d, window)
    moth_to_fire = standardize(a_swing) + standardize(jump)
    return moth_to_fire

class MinFactorCal:
    @property
    def trade_days_all(self):
        return pd.read_csv(os.path.join(fdir, 'raw_data', 'trade_days_all.csv'), 
                           index_col=[0], parse_dates=True)
    
    def get_dailydata(self, fname):
        return pd.read_csv(os.path.join(fdir, 'raw_data', 'dailydata', f'{fname}.csv'), 
                           index_col=[0], parse_dates=True)
    
    def get_mindata(self, sdate, edate, stime=None, etime=None, 
                    columns=None, callback=None, dailydfs=None): 
        sidx, eidx = (get_index(sdate, self.trade_days_all.index), 
                      get_index(edate, self.trade_days_all.index))
        tdays = self.trade_days_all.index[sidx:eidx+1]
                
        fnames = [td.strftime('%Y%m%d')+'.pq' for td in tdays]
        if dailydfs is not None:
            dfs = {dt: {dayfname: dailydfs[dayfname].loc[dt] for dayfname in dailydfs.keys()} 
                   for dt in tdays}
        else:
            dfs = {}
            
        fdata = []
        for fname in fnames:
            dt = pd.to_datetime(fname.split('.')[0])
            tmp = open_mintable(fname, columns=columns, stime=stime, etime=etime,
                                callback=callback, **dfs.get(dt, {}))
            fdata.append(tmp)
            print(fname)
        oridata = pd.concat(fdata)
        oridata = oridata.set_index(['tradedate'], append=True)
        oridata = oridata.swaplevel(0,1).sort_index()
        return oridata
    
    def get_mothtofire(self, sdate, edate, window):
        adjf = self.get_dailydata('adjfactor') #后复权因子
        high = self.get_dailydata('highprice')
        low = self.get_dailydata('lowprice')
        close = self.get_dailydata('closeprice')
        
        jump_d = self.get_mindata(sdate, edate, columns=['close'],
                                  callback=jump)
        jump_d = jump_d.squeeze().unstack()
        
        mtf = moth_to_fire(jump_d, high*adjf, low*adjf, close*adjf, window)
        return mtf
    
if __name__ == '__main__':
    minfac = MinFactorCal()
    window = 20
    calcols = ['close']
    
    sdate = '2017-1-3'; edate = '2017-2-7'
    mtf = minfac.get_mothtofire(sdate, edate, window)
    print(mtf)
    