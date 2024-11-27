# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:23:28 2024

@author: hazc
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from utils import get_index, open_mintable, js, bns, iv_hat
import warnings
warnings.filterwarnings("ignore")

fdir = r'E:\跳跃因子'

def jump_gf1(df, resample_periods=5):
    close = df['close'].unstack()
    close_resam = close.resample(f'{resample_periods}T', label='right', closed='right')
    close_resam = close_resam.last()
    ret_resam = close_resam.pct_change()
    
    drop_ts = ret_resam.index.indexer_between_time('11:31:00', '13:00:00')   
    save_ts = ret_resam.index.difference(ret_resam.index[drop_ts])
    ret_resam = ret_resam.loc[save_ts].dropna(how='all', axis=0)
    
    ret_resam_log = np.log(ret_resam + 1).sort_index()
    ret_log = ret_resam_log.values.copy()
    
    rv = np.nansum(ret_log ** 2, axis=0)
    iv = iv_hat(ret_log)
    rjv = np.where(rv - iv > 0, rv - iv, 0)
    
    rv_pos = np.nansum((ret_log ** 2) * (ret_log > 0), axis=0)
    rv_neg = np.nansum((ret_log ** 2) * (ret_log < 0), axis=0)
    sj = rv_pos - rv_neg
    # rsj = sj / rv
    
    rjvp = np.where(rv_pos - iv/2 > 0, rv_pos - iv/2, 0)
    rjvn = np.where(rv_neg - iv/2 > 0, rv_neg - iv/2, 0)
    srjv = rjvp - rjvn
    
    thres = 4 * (ret_log.shape[0]) ** (-0.49) * np.sqrt(iv)
    rv_thres = np.nansum((ret_log ** 2) * (np.abs(ret_log) > thres), axis=0)
    rljv = np.where(rjv < rv_thres, rjv, rv_thres)
    rsjv = rjv - rljv
    
    rv_pos_thres = np.nansum((ret_log ** 2) * (ret_log >= thres), axis=0)
    rv_neg_thres = np.nansum((ret_log ** 2) * (ret_log <= -thres), axis=0)
    rljvp = np.where(rjvp < rv_pos_thres, rjvp, rv_pos_thres)
    rljvn = np.where(rjvn < rv_neg_thres, rjvn, rv_neg_thres)
    srljv = rljvp - rljvn
    
    rsjvp = rjvp - rljvp
    rsjvn = rjvn - rljvn
    srsjv = rsjvp - rsjvn
    
    res = np.r_[[rjv, sj, rjvp, rjvn, srjv, rljv, rsjv, 
                 rljvp, rljvn, srljv, rsjvp, rsjvn, srsjv]]
    res = pd.DataFrame(res, index=['rjv', 'sj', 'rjvp', 'rjvn', 'srjv', 'rljv', 'rsjv', 
                                   'rljvp', 'rljvn', 'srljv', 'rsjvp', 'rsjvn', 'srsjv'],
                       columns=close.columns)
    res_div_rv = res / rv
    res_div_rv.index = ['r_'+idx for idx in res_div_rv.index]
    res = pd.concat([res, res_div_rv])
    res = res.T.assign(tradedate=close.index[0].date())
    return res

def jump_gf2(df, resample_periods=5, p=0.05): 
    close = df['close'].unstack()
    close_resam = close.resample(f'{resample_periods}T', label='right', closed='right')
    close_resam = close_resam.last()
    ret_resam = close_resam.pct_change()
    
    drop_ts = ret_resam.index.indexer_between_time('11:31:00', '13:00:00')   
    save_ts = ret_resam.index.difference(ret_resam.index[drop_ts])
    ret_resam = ret_resam.loc[save_ts].dropna(how='all', axis=0)
    
    ret_resam_ori = ret_resam.sort_index()
    ret_resam_log = np.log(ret_resam + 1).sort_index()
    
    ret_ori, ret_log = ret_resam_ori.values.copy(), ret_resam_log.values.copy()
    thres = stats.norm.ppf(1 - p/2)
    
    jump_bns = bns(ret_log)
    I_jump_bns = jump_bns > thres
    jump_js = js(ret_ori, ret_log, p=4)
    I_jump_js = jump_js > thres
    
    I_jump_bns = I_jump_bns.astype(float)
    I_jump_js = I_jump_js.astype(float)
    
    res = np.c_[jump_bns, jump_js, I_jump_bns, I_jump_js]
    res = pd.DataFrame(res, index=close.columns, 
                       columns=['jump_bns', 'jump_js', 'i_jump_bns', 'i_jump_js'])
    res = res.assign(tradedate=close.index[0].date())
    return res

def jump_gf2_day(datdf, pc, window=20, p=0.05):
    i_jump_bns = datdf['i_jump_bns'].unstack(level='wind_code')
    i_jump_js = datdf['i_jump_js'].unstack(level='wind_code')
    jarr_bns = i_jump_bns.rolling(window).mean()
    jarr_js = i_jump_js.rolling(window).mean()
    jarr_pos_bns = (i_jump_bns * (pc > 0)).rolling(window).mean()
    jarr_neg_bns = (i_jump_bns * (pc < 0)).rolling(window).mean()
    jarr_pos_js = (i_jump_js * (pc > 0)).rolling(window).mean()
    jarr_neg_js = (i_jump_js * (pc < 0)).rolling(window).mean()
    
    jr_bns = np.exp((i_jump_bns * pc).rolling(window).sum()) - 1
    jr_js = np.exp((i_jump_js * pc).rolling(window).sum()) - 1
    jr_pos_bns = np.exp((i_jump_bns * pc.mask(pc<=0, 0)).rolling(window).sum()) - 1
    jr_neg_bns = np.exp((i_jump_bns * pc.mask(pc>=0, 0)).rolling(window).sum()) - 1
    jr_pos_js = np.exp((i_jump_js * pc.mask(pc<=0, 0)).rolling(window).sum()) - 1
    jr_neg_js = np.exp((i_jump_js * pc.mask(pc>=0, 0)).rolling(window).sum()) - 1
    
    jar_bns = np.exp((i_jump_bns * pc.abs()).rolling(window).sum()) - 1
    jar_js = np.exp((i_jump_js * pc.abs()).rolling(window).sum()) - 1
    jar_pos_bns = np.exp((i_jump_bns * pc.abs().mask(pc<=0, 0)).rolling(window).sum()) - 1
    jar_neg_bns = np.exp((i_jump_bns * pc.abs().mask(pc>=0, 0)).rolling(window).sum()) - 1
    jar_pos_js = np.exp((i_jump_js * pc.abs().mask(pc<=0, 0)).rolling(window).sum()) - 1
    jar_neg_js = np.exp((i_jump_js * pc.abs().mask(pc>=0, 0)).rolling(window).sum()) - 1
    
    jump_bns = datdf['jump_bns'].unstack(level='wind_code')
    jump_js = datdf['jump_js'].unstack(level='wind_code')
    jt_avg_bns = jump_bns.rolling(window).mean()
    jt_avg_js = jump_js.rolling(window).mean()

    thres = stats.norm.ppf(1 - p/2)    
    srjv = datdf['srjv'].unstack(level='wind_code')
    rjvp = datdf['rjvp'].unstack(level='wind_code')
    tsrjv_bns = (jump_bns * srjv / thres).rolling(window).sum() / \
                (jump_bns / thres).rolling(window).sum()
    tsrjv_js = (jump_js * srjv / thres).rolling(window).sum() / \
                (jump_js / thres).rolling(window).sum()
                
    tcjv_bns = np.where(jump_bns / thres >= 1, srjv, np.where(jump_bns / thres < 1, rjvp, np.nan))
    tcjv_js = np.where(jump_js / thres >= 1, srjv, np.where(jump_js / thres < 1, rjvp, np.nan))
    tcjv_bns = pd.DataFrame(tcjv_bns, index=jump_bns.index, columns=jump_bns.columns)
    tcjv_js = pd.DataFrame(tcjv_js, index=jump_js.index, columns=jump_js.columns)
    tsrjvp_bns = tcjv_bns.rolling(window).mean()
    tsrjvp_js = tcjv_js.rolling(window).mean()
    
    res = {'jarr_bns': jarr_bns, 'jarr_pos_bns': jarr_pos_bns, 'jarr_neg_bns': jarr_neg_bns,
           'jarr_js': jarr_js,  'jarr_pos_js': jarr_pos_js, 'jarr_neg_js': jarr_neg_js, 
           'jr_bns': jr_bns, 'jr_pos_bns': jr_pos_bns, 'jr_neg_bns': jr_neg_bns,
           'jr_js': jr_js, 'jr_pos_js': jr_pos_js, 'jr_neg_js': jr_neg_js,
           'jar_bns': jar_bns, 'jar_pos_bns': jar_pos_bns, 'jar_neg_bns': jar_neg_bns,
           'jar_js': jar_js, 'jar_pos_js': jar_pos_js, 'jar_neg_js': jar_neg_js,
           'jt_avg_bns': jt_avg_bns, 'jt_avg_js': jt_avg_js,
           'tsrjv_bns': tsrjv_bns, 'tsrjv_js': tsrjv_js,
           'tsrjvp_bns': tsrjvp_bns, 'tsrjvp_js': tsrjvp_js}
    return res

class MinFactorCal:
    def __init__(self):
        self.factor = {}
    
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
    
    def get_jump1(self, sdate, edate):
        jump1 = self.get_mindata(sdate, edate, columns=['close'],
                                 callback=jump_gf1)
        self.factor['jump1'] = jump1
        return jump1
    
    def get_jump2(self, sdate, edate, window):
        pc = self.get_dailydata('changepct')
        
        if 'jump1' not in self.factor:
            jump1 = self.get_mindata(sdate, edate, columns=['close'],
                                     callback=jump_gf1)
        else:
            jump1 = self.factor['jump1']
        jump2 = self.get_mindata(sdate, edate, columns=['close'],
                                 callback=jump_gf2)
        datdf = pd.concat([jump1[['srjv','rjvp']], jump2], axis=1)
        jump2_day = jump_gf2_day(datdf, pc, window)
        return jump2_day
    
if __name__ == '__main__':
    minfac = MinFactorCal()
    window = 20
    calcols = ['close']
    
    sdate = '2017-1-3'; edate = '2017-2-7'
    jump1 = minfac.get_jump1(sdate, edate)
    print(jump1)
    jump2_day = minfac.get_jump2(sdate, edate, window)
    print(jump2_day)
    