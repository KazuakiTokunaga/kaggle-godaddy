import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from modules import utils

mbd = 'microbusiness_density'


def add_lag_features(df_all, max_scale=40, USE_LAG=7):
    print(f'add lag features: max_scale={max_scale}')

    for i in range(30, max_scale+1):
        dt = df_all.loc[df_all.scale==i].groupby('cfips')['active'].agg('last')
        df_all[f'select_lastactive{i}'] = df_all['cfips'].map(dt).astype(float)

        dt = df_all.loc[df_all.scale==i].groupby('cfips')[mbd].agg('last')
        df_all[f'select_lastmbd{i}'] = df_all['cfips'].map(dt).astype(float)

    indices = df_all['cfips'].isin([28055, 48301]) # cfips having zero mbd
    df_t = df_all[~indices].copy()
    df_t2 = df_all[indices].copy()
    df_t[f'select_rate1'] = df_t.groupby('cfips')[mbd].shift(1).bfill()
    df_t[f'select_rate1'] = (df_t[mbd] / df_t[f'select_rate1'] - 1).fillna(0)
    df_t2[f'select_rate1'] = 0
    df_all = pd.concat([df_t, df_t2])

    for i in range(1, USE_LAG+1):
        df_all[f'select_active_lag{i}'] = df_all.groupby('cfips')['active'].shift(i).bfill()
        df_all[f'select_mbd_lag{i}'] = df_all.groupby('cfips')[mbd].shift(i).bfill()

    for k in range(1, USE_LAG+1):
        df_all[f'select_active_lag1_diff{k}'] = df_all.groupby('cfips')[f'select_active_lag1'].diff(k)

    for i in range(1, USE_LAG+1):
        df_all[f'select_rate1_lag{i}'] = df_all[f'select_rate1'].shift(i).bfill()

    for c in [2,4,6,8,10]:
        df_all[f'select_rate1_rsum{c}'] = df_all.groupby('cfips')[f'select_rate1_lag1'].transform(lambda s: s.rolling(c, min_periods=1).sum())   

    return df_all


def create_features(df_all, pred_m, train_times, USE_LAG = 5):
    drop_features = [mbd, 'state', 'active', 'county', 'cfips', 'month', 'year']
    features = list(filter(lambda x: (not x.startswith('select_') and (x not in drop_features)),  df_all.columns.to_list()))
    features += list(filter(lambda x: (x.startswith(f'select_rate{pred_m}_')), df_all.columns.to_list()))
    features += list(filter(lambda x: (x.startswith(f'select_active_lag{pred_m}_diff')), df_all.columns.to_list()))
    
    return features


def get_trend_dict(df_all, train_time=40, n=3, thre=3, active_thre=25000, 
                    regularize=True, v_regularize=0.003, v_clip=[0.995, 1.005]):

    dt = df_all.loc[df_all.scale==train_time].groupby('cfips')['active'].agg('last')
    df_all[f'select_lastactive{train_time}'] = df_all['cfips'].map(dt).astype(float)

    idx = (df_all['scale']>= train_time-n)&(df_all['scale']<=train_time)&(df_all[f'select_lastactive{train_time}']>=active_thre)
    df_target_lag = df_all[idx].copy()
    for i in range(1, n+1):
        df_target_lag[f'lag_{i}'] = df_target_lag[mbd].shift(i)

    for i in range(1, n+1):
        if i==1:
            df_target_lag[f'rate{i}'] = df_target_lag[mbd] / df_target_lag[f'lag_{i}']
        else:
            df_target_lag[f'rate{i}'] = df_target_lag[f'lag_{i-1}'] / df_target_lag[f'lag_{i}']

    cs = ['cfips', mbd, 'active', 'scale'] + [f'rate{i}' for i in range(1, n+1)]
    df_target = df_target_lag.loc[df_target_lag['scale']==train_time, cs].copy()

    df_target['up_cnt'] = 0
    df_target['down_cnt'] = 0
    df_target['mean'] = 0
    for i in range(1, n+1):
        df_target['up_cnt'] += (df_target[f'rate{i}'] > 1)*1
        df_target['down_cnt'] += (df_target[f'rate{i}']<1)*1
        df_target['mean'] += df_target[f'rate{i}']
    df_target['mean'] /= n

    df_target['trend'] = df_target[['up_cnt', 'mean']].apply(lambda x: x[1] if x[0] >= thre and x[1]>1 else np.nan, axis=1)
    df_target['trend'] = df_target[['down_cnt', 'mean', 'trend']].apply(lambda x: x[1] if x[0] >= thre and x[1]<1 else x[2], axis=1)
    
    if regularize:
        df_target['trend'] = df_target['trend'].apply(utils.regularize, v=v_regularize)

    df_trend = df_target[~df_target['trend'].isna()].copy()

    df_trend['trend'] = df_trend['trend'].clip(v_clip[0], v_clip[1])
    trend_dict = df_trend[['cfips', 'trend']].set_index('cfips').to_dict()['trend']
    
    return df_trend, trend_dict


def get_trend_multi(df_all, train_time=40, m_len=5, upper_bound=140, lower_bound=20, multi_can=[1.00, 1.002, 1.004]):

    df_extract = df_all[(df_all['scale']<=train_time)&(df_all['scale']>=train_time-m_len)].copy()
    df_multi = df_extract[(df_extract[f'select_lastactive{train_time}']<=upper_bound)&(df_extract[f'select_lastactive{train_time}']>=lower_bound)].copy()

    mult_column_to_mult = {f'smape_{mult}': mult for mult in multi_can}

    for mult_column, mult in mult_column_to_mult.items():
        df_multi['y_pred'] = df_multi['select_mbd_lag1'] * mult
        df_multi[mult_column] = utils.smape_arr(df_multi[mbd], df_multi['y_pred'])
        
    df_agg = df_multi.groupby('cfips')[list(mult_column_to_mult.keys())].mean().copy()
    df_agg['best_mult'] = df_agg.idxmin(axis=1).map(mult_column_to_mult)

    cfips_to_best_mult = dict(zip(df_agg.index, df_agg['best_mult']))

    return cfips_to_best_mult
