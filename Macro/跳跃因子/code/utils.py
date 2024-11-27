# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:36:32 2024

@author: hazc
"""
import os
import numpy as np
import numba as nb
import pandas as pd
from functools import reduce

fdir = r'E:\跳跃因子'

@nb.jit
def weighted_average(df, wt):
    wt = (~np.isnan(df)) * wt
    return np.nansum(wt * df) / np.sum(wt)

def get_exp_weight(window, half_life):
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] 

def numpy_shift(matrix, shift_by, fill_value=np.nan):
    if shift_by == 0:
        return matrix
    elif shift_by > 0:
        return np.vstack((np.full((shift_by, matrix.shape[1]), fill_value), matrix[:-shift_by]))
    else:
        return np.vstack((matrix[-shift_by:], np.full((-shift_by, matrix.shape[1]), fill_value)))

def standardize(data, weight_df=None):
    df = data.copy()
    cal_axis = 1 if isinstance(df, pd.DataFrame) else 0
    if weight_df is None:
        mean = df.mean(axis=cal_axis) 
    else:
        wt_sum = weight_df.sum(axis=cal_axis) 
        mean = (df * weight_df).sum(axis=cal_axis) / wt_sum
    std = df.std(axis=cal_axis, ddof=1)
    df_stan = df.sub(mean, axis=0).div(std, axis=0)
    return df_stan

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

def check_same_matrix(mat1, mat2):
    return (np.where(np.isnan(mat1), np.inf, mat1) == np.where(np.isnan(mat2), np.inf, mat2)).all()

def miu(p):
    return 2**(p/2) * np.math.gamma((p+1)/2) / np.sqrt(np.math.pi)

def iv_hat(ret_df, m=2/3):
    sum_prod = np.nansum((np.abs(numpy_shift(ret_df, -2)) ** m) * \
                         (np.abs(numpy_shift(ret_df, -1)) ** m) * \
                         (np.abs(ret_df) ** m), axis=0) 
    iv = miu(m) ** (-2/m) * sum_prod
    return iv

def js(ret_ori, ret_log):
    n = ret_ori.shape[0]
    rv = np.sum(ret_log ** 2, axis=0)
    swv = 2 * np.sum(ret_ori - ret_log, axis=0)
    v_hat = 1 / miu(1) ** 2 * np.nansum(np.abs(numpy_shift(ret_log, -1)) * np.abs(ret_log), axis=0) #ret_log.shift(-1).abs() * ret_log.abs()
    sum_prod = np.nansum(reduce(np.multiply, [np.abs(numpy_shift(ret_log, -i-1)) for i in range(6)]), axis=0) #ret_log.shift(-i-1).abs()
    sigma_swv = miu(6) / 9 * (n ** 3) * (miu(1) ** (-6)) / (n - 5) * sum_prod
    js = n * v_hat / np.sqrt(sigma_swv) * (1 - rv / swv)
    return js

def bns(ret_log):
    pi = np.math.pi
    n = ret_log.shape[0]
    rv = np.nansum(ret_log ** 2, axis=0)
    bv = miu(1) ** (-2) * n/(n-1) * np.nansum(np.abs(ret_log) * \
                                               np.abs(numpy_shift(ret_log, -1)), axis=0)
    sum_prod = np.nansum((np.abs(numpy_shift(ret_log, -2)) ** (4/3)) * \
                         (np.abs(numpy_shift(ret_log, -1)) ** (4/3)) * \
                         (np.abs(ret_log) ** (4/3)), axis=0) 
    tp = miu(4/3) ** (-3) * (n**2) / (n-2) * sum_prod
    bns = (1 - bv / rv) / np.sqrt(((pi/2)**2 + pi - 5) / n * \
                                  np.where(tp / bv**2 > 1, tp / bv**2, 1))
    return bns