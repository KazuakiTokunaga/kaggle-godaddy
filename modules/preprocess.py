import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from modules import utils

mbd = 'microbusiness_density'


def add_lag_features(df_all, max_scale=38, USE_LAG=5, seasonal=False):
    print(f'add lag features: max_scale={max_scale}')

    for i in range(30, max_scale+1):
        dt = df_all.loc[df_all.scale==i].groupby('cfips')['active'].agg('last')
        df_all[f'select_lastactive{i}'] = df_all['cfips'].map(dt).astype(float)

        dt = df_all.loc[df_all.scale==i].groupby('cfips')[mbd].agg('last')
        df_all[f'select_lastmbd{i}'] = df_all['cfips'].map(dt).astype(float)
    
    for i in range(1, 5):
        df_all.loc[df_all['cfips']==28055, mbd] = 1

        df_all[f'select_rate{i}'] = df_all.groupby('cfips')[mbd].shift(i).bfill()
        df_all[f'select_rate{i}'] = (df_all[mbd] / df_all[f'select_rate{i}'] - 1).fillna(0)

        df_all.loc[df_all['cfips']==28055, mbd] = 0

    for i in range(1, 4+USE_LAG):
        df_all[f'select_active_lag{i}'] = df_all.groupby('cfips')['active'].shift(i).bfill()
        df_all[f'select_mbd_lag{i}'] = df_all.groupby('cfips')[mbd].shift(i).bfill()


    for i in range(1, 5):
        for j in range(i, i + USE_LAG):
            df_all[f'select_rate{i}_lag{j}'] = df_all[f'select_rate{i}'].shift(j).bfill()

    for i in range(1, 5):
        for c in [k for k in range(3, USE_LAG+1)]:
            df_all[f'select_rate{i}_rsum{c}'] = 0
            for k in range(i, i+c):
                df_all[f'select_rate{i}_rsum{c}'] += df_all[f'select_rate{i}_lag{k}']

    return df_all


def create_features(df_all, pred_m, train_times, USE_LAG = 5):
    drop_features = [mbd, 'active', 'scale']
    features = list(filter(lambda x: (not x.startswith('select_') and (x not in drop_features)),  df_all.columns.to_list()))
    
    # Select appropriate lastactive and lastmbd features.
    features.append(f'select_lastactive{train_times}')
    features.append(f'select_lastmbd{train_times}')
    features += list(filter(lambda x: (x.startswith(f'select_rate{pred_m}_')), df_all.columns.to_list()))
    
    # Select appropriate target and lag features.
    for i in range(pred_m, pred_m + USE_LAG):
        features.append(f'select_active_lag{i}')
        features.append(f'select_mbd_lag{i}')
    
    return features


def get_trend_dict(df_all, train_time=37, n=3, thre=3, active_thre=25000, 
                    regularize=True, v_regularize=0.003, v_clip=[0.995, 1.005]):

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