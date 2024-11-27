# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:00:56 2024

@author: hazc
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from utils import (get_index, open_mintable, get_exp_weight, 
                   check_same_matrix, weighted_average, js)
import warnings
warnings.filterwarnings("ignore")

fdir = r'E:\跳跃因子'

def jump_ret(datdf, turnover, window=20):
    min_periods = half_life = window // 2
    if not datdf.index.names[0] == 'tradedate':
        datdf = datdf.swaplevel(0,1).sort_index()
    jump = datdf['jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    no_jump = datdf['non_jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    pos_jump = datdf['pos_jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    neg_jump = datdf['neg_jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    co_jump = datdf['ovn_jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    oc_jump = datdf['intra_jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    co_pos_jump = datdf['ovn_pos_jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    oc_pos_jump = datdf['intra_pos_jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    co_neg_jump = datdf['ovn_neg_jump_ret_fac'].unstack().rolling(window, min_periods).sum()
    oc_neg_jump = datdf['intra_neg_jump_ret_fac'].unstack().rolling(window, min_periods).sum()

    oc_pos_jump_std = datdf['intra_pos_jump_ret_fac'].unstack().rolling(window, min_periods).std()
    weight = get_exp_weight(window, half_life)
    oc_pos_jump_avg = datdf['intra_pos_jump_ret_fac'].unstack().rolling(window).\
                        apply(weighted_average, args=(weight,))
    oc_pos_jumpt_ret = datdf['intra_pos_jump_ret_fac'].unstack() * turnover.reindex(index=datdf.index.levels[0], 
                                                                                    columns=datdf.index.levels[1])
    oc_pos_jumpt_avg = oc_pos_jumpt_ret.rolling(window).apply(weighted_average, args=(weight,))
    
    res_all = pd.concat([jump.unstack(), no_jump.unstack(), pos_jump.unstack(), 
                         neg_jump.unstack(), co_jump.unstack(), oc_jump.unstack(),
                         co_pos_jump.unstack(), oc_pos_jump.unstack(), co_neg_jump.unstack(),
                         oc_neg_jump.unstack(), oc_pos_jump_std.unstack(), 
                         oc_pos_jump_avg.unstack(), oc_pos_jumpt_avg.unstack()], axis=1)
    res_all.columns = ['jump', 'no_jump', 'pos_jump', 'neg_jump', 'co_jump', 'oc_jump',
                       'co_pos_jump', 'oc_pos_jump', 'co_neg_jump', 'oc_neg_jump',
                       'oc_pos_jump_std', 'oc_pos_jump_avg', 'oc_pos_jumpt_avg']
    return res_all.dropna(how='all', axis=0)
    
def jump(df, overnightret, resample_periods=5, p=0.05):
    close = df['close'].unstack(); cdate = close.index[0].date()
    close_resam = close.resample(f'{resample_periods}T', label='right', closed='right')
    close_resam = close_resam.last()
    ret_resam = close_resam.pct_change()
    
    drop_ts = ret_resam.index.indexer_between_time('11:31:00', '13:00:00')   
    save_ts = ret_resam.index.difference(ret_resam.index[drop_ts])
    ret_resam = ret_resam.loc[save_ts].dropna(how='all', axis=0)
    
    overnight_timeidx = pd.to_datetime(f'{cdate.strftime("%Y-%m-%d")} 09:30:00')
    # overnightret = overnightret_all.loc[cdate.strftime('%Y%m%d')]
    ret_resam.loc[overnight_timeidx] = overnightret.reindex(ret_resam.columns)
    ret_resam_ori = ret_resam.sort_index()
    ret_resam_log = np.log(ret_resam + 1).sort_index()
    
    ret_ori, ret_log = ret_resam_ori.values.copy(), ret_resam_log.values.copy()
    jump_time = get_jump_time(ret_ori, ret_log, p=p)
    jump_time = np.where(jump_time == 0, np.nan, jump_time)
    
    jump_ret = ret_resam_log * jump_time
    jump_ret_fac = jump_ret.sum(axis=0)
    non_jump_ret_fac = ret_resam_log.sum(axis=0) - jump_ret_fac
    
    pos_jump_ret_fac = jump_ret.mask(jump_ret<0).sum(axis=0)
    neg_jump_ret_fac = jump_ret.mask(jump_ret>0).sum(axis=0)
    
    ovn_jump_ret, intra_jump_ret = jump_ret.iloc[0], jump_ret.iloc[1:]
    ovn_jump_ret_fac = ovn_jump_ret.fillna(0)
    intra_jump_ret_fac = intra_jump_ret.sum(axis=0)
    ovn_pos_jump_ret_fac = ovn_jump_ret.mask(ovn_jump_ret<0, np.nan).fillna(0)
    ovn_neg_jump_ret_fac = ovn_jump_ret.mask(ovn_jump_ret>0, np.nan).fillna(0)
    intra_pos_jump_ret_fac = intra_jump_ret.mask(intra_jump_ret<0, np.nan).sum(axis=0)
    intra_neg_jump_ret_fac = intra_jump_ret.mask(intra_jump_ret>0, np.nan).sum(axis=0)

    res = pd.concat([jump_ret_fac, non_jump_ret_fac, pos_jump_ret_fac, neg_jump_ret_fac,
                     intra_jump_ret_fac, intra_pos_jump_ret_fac, intra_neg_jump_ret_fac,
                     ovn_jump_ret_fac, ovn_pos_jump_ret_fac, ovn_neg_jump_ret_fac], axis=1)
    res.columns = ['jump_ret_fac', 'non_jump_ret_fac', 'pos_jump_ret_fac', 'neg_jump_ret_fac',
                   'intra_jump_ret_fac', 'intra_pos_jump_ret_fac', 'intra_neg_jump_ret_fac',
                   'ovn_jump_ret_fac', 'ovn_pos_jump_ret_fac', 'ovn_neg_jump_ret_fac']
    res = res.assign(tradedate=cdate)
    return res

def get_jump_time(ret_ori, ret_log, jump_time=None, p=0.05):
    if jump_time is None:
        jump_time = np.zeros(ret_ori.shape)
    col_idx = np.arange(ret_ori.shape[1])

    js_day = js(ret_ori, ret_log)
    thres = stats.norm.ppf(1 - p/2)
    jump_time[:, (np.abs(js_day) < thres)] = np.where(jump_time[:, (np.abs(js_day) < thres)] == 0, np.nan, 
                                                      jump_time[:, (np.abs(js_day) < thres)]) 
    if (np.abs(js_day) < thres).all():
        return jump_time
    js_day[(np.abs(js_day) < thres)] = np.nan
    jump_time_ori = jump_time.copy()
    
    delta = []
    for j in range(ret_ori.shape[0]):
        ret_ori_tmp = ret_ori.copy(); ret_log_tmp = ret_log.copy()
        ret_ori_tmp[j, :] = np.nanmedian(ret_ori, axis=0)
        ret_log_tmp[j, :] = np.nanmedian(ret_log, axis=0)
        
        js_day_med = js(ret_ori_tmp, ret_log_tmp)
        js_day_delta = np.abs(js_day) - np.abs(js_day_med)
        delta.append(js_day_delta)
        
    delta = np.r_[delta]
    jump_ridx = np.argmax(delta, axis=0)
    valid_jump_ridx = jump_ridx[~np.isnan(js_day)]
    valid_col_idx = col_idx[~np.isnan(js_day)]
    jump_time[valid_jump_ridx, valid_col_idx] = np.where(jump_time[valid_jump_ridx, valid_col_idx]==0, 1, 
                                                         jump_time[valid_jump_ridx, valid_col_idx])
    if check_same_matrix(jump_time_ori, jump_time):
        return jump_time
    
    ret_ori[valid_jump_ridx, valid_col_idx] = np.nanmedian(ret_ori, axis=0)[~np.isnan(js_day)]
    ret_log[valid_jump_ridx, valid_col_idx] = np.nanmedian(ret_log, axis=0)[~np.isnan(js_day)]
    return get_jump_time(ret_ori, ret_log, jump_time, p=p)

class MinFactorCal:
    def get_dailydata(self, fname):
        return pd.read_csv(os.path.join(fdir, 'raw_data', 'dailydata', f'{fname}.csv'), 
                           index_col=[0], parse_dates=True)
        #turnoverrate, closeprice, prevcloseprice    
    
    @property
    def trade_days_all(self):
        return pd.read_csv(os.path.join(fdir, 'raw_data', 'trade_days_all.csv'), 
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
                                callback=callback, **dfs.get(dt))
            fdata.append(tmp)
            print(fname)
        oridata = pd.concat(fdata)
        oridata = oridata.set_index(['tradedate'], append=True)
        oridata = oridata.swaplevel(0,1).sort_index()
        return oridata
    
    def get_jump(self, sdate, edate, window=20):
        open_ = self.get_dailydata('openprice')
        prevclose = self.get_dailydata('prevcloseprice')
        overnightret = open_ / prevclose - 1

        jump_daily_df = self.get_mindata(sdate, edate, columns=['close'], 
                                         callback=jump, dailydfs={'overnightret': overnightret})
        turnover = self.get_dailydata('turnoverrate')
        jump_factor = jump_ret(jump_daily_df, turnover, window)
        return jump_factor
    
if __name__ == '__main__':
    minfac = MinFactorCal()
    window = 20
    calcols = ['close']
    
    sdate = '2017-1-3'; edate = '2017-2-7'
    jump_factor = minfac.get_jump(sdate, edate, window)
    print(jump_factor)
    