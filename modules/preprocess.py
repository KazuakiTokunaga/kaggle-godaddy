import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from modules import utils

mbd = 'microbusiness_density'


def add_lag_features(df_all, max_scale=40, USE_LAG=7, max_window=12):
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
        df_all[f'select_mbd_origin_lag{i}'] = df_all.groupby('cfips')['mbd_origin'].shift(i).bfill()

    for k in range(1, USE_LAG+1):
        df_all[f'select_active_lag1_diff{k}'] = df_all.groupby('cfips')[f'select_active_lag1'].diff(k)

    for i in range(1, USE_LAG+1):
        df_all[f'select_rate1_lag{i}'] = df_all[f'select_rate1'].shift(i).bfill()

    for c in [k for k in range(2, max_window+1, 2)]:
        df_all[f'select_rate1_rsum{c}'] = df_all.groupby('cfips')[f'select_rate1_lag1'].transform(lambda s: s.rolling(c, min_periods=1).sum())   

    return df_all


def create_features(df_all, pred_m, train_times, USE_LAG = 5):
    drop_features = [mbd, 'mbd_origin', 'state', 'active', 'county', 'cfips', 'month', 'year']
    features = list(filter(lambda x: (not x.startswith('select_') and (x not in drop_features)),  df_all.columns.to_list()))
    features += list(filter(lambda x: (x.startswith(f'select_rate{pred_m}_')), df_all.columns.to_list()))
    features += list(filter(lambda x: (x.startswith(f'select_active_lag{pred_m}_diff')), df_all.columns.to_list()))
    
    return features

def regularize(x, v):
    if x >= 1:
        if x * (1-v[1]) > 1:
            x *= (1-v[1])
        else:
            x = None
    else:
        if x * (1+v[0]) < 1:
            x *= (1+v[0])
        else:
            x = None
    return x


def get_trend_dict(df_all, train_time=40, n=2, thre=2, thre_r = 0.002, lower_bound=20, upper_bound=140, 
                    use_regularize=True, v_regularize=[0.025, 0.025], v_clip=[0.9975, 1.004]):

    dt = df_all.loc[df_all.scale==train_time].groupby('cfips')['active'].agg('last')
    df_all[f'select_lastactive{train_time}'] = df_all['cfips'].map(dt).astype(float)

    idx = (df_all['scale']>= train_time-n)&(df_all['scale']<=train_time)&(df_all[f'select_lastactive{train_time}']>lower_bound)&(df_all[f'select_lastactive{train_time}']<=upper_bound)
    df_target_lag = df_all[idx].copy()
    for i in range(1, n+1):
        df_target_lag[f'lag_{i}'] = df_target_lag['mbd_origin'].shift(i)

    for i in range(1, n+1):
        if i==1:
            df_target_lag[f'rate{i}'] = df_target_lag['mbd_origin'] / df_target_lag[f'lag_{i}']
        else:
            df_target_lag[f'rate{i}'] = df_target_lag[f'lag_{i-1}'] / df_target_lag[f'lag_{i}']

    cs = ['cfips', 'mbd_origin', 'active', 'scale'] + [f'rate{i}' for i in range(1, n+1)]
    df_target = df_target_lag.loc[df_target_lag['scale']==train_time, cs].copy()

    df_target['up_cnt'] = 0
    df_target['down_cnt'] = 0
    df_target['mean'] = 0
    for i in range(1, n+1):
        df_target['up_cnt'] += (df_target[f'rate{i}'] > (1+thre_r))*1
        df_target['down_cnt'] += (df_target[f'rate{i}']< (1-thre_r))*1
        df_target['mean'] += df_target[f'rate{i}']
    df_target['mean'] /= n

    df_target['trend'] = df_target[['up_cnt', 'mean']].apply(lambda x: x[1] if x[0] >= thre and x[1]>1 else np.nan, axis=1)
    df_target['trend'] = df_target[['down_cnt', 'mean', 'trend']].apply(lambda x: x[1] if x[0] >= thre and x[1]<1 else x[2], axis=1)
    
    if regularize:
        df_target['trend'] = df_target['trend'].apply(regularize, v=v_regularize)

    df_trend = df_target[~df_target['trend'].isna()].copy()

    df_trend['trend'] = df_trend['trend'].clip(v_clip[0], v_clip[1])
    trend_dict = df_trend[['cfips', 'trend']].set_index('cfips').to_dict()['trend']
    trend_dict = {k: v for k, v in trend_dict.items() if v != 1.0}
    
    return df_trend, trend_dict


def rot(df):
    for angle in [15, 30, 45]:
        df[f'rot_{angle}_x'] = (np.cos(np.radians(angle)) * df['lat']) + \
                                (np.sin(np.radians(angle)) * df['lng'])
        
        df[f'rot_{angle}_y'] = (np.cos(np.radians(angle)) * df['lat']) - \
                                (np.sin(np.radians(angle)) * df['lng'])
        
    return df


def add_location(df_all, use_umap=False, version='v2'):

    if version=='v1':

        # import reverse_geocoder as rg
        # from sklearn.preprocessing import LabelEncoder
        # from sklearn.decomposition import PCA
        # from sklearn.cluster import KMeans

        coordinates = df_all[['lng', 'lat']].values

        # Encoding tricks
        emb_size = 20
        precision = 1e6

        latlon = np.expand_dims(coordinates, axis=-1)

        m = np.exp(np.log(precision)/emb_size)
        angle_freq = m ** np.arange(emb_size)
        angle_freq = angle_freq.reshape(1,1, emb_size)
        latlon = latlon * angle_freq
        latlon[..., 0::2] = np.cos(latlon[..., 0::2])

        coordinates = list(zip(df_all['lat'], df_all['lng']))
        results = rg.search(coordinates)
        df_all['place'] = [x['admin2'] for x in results]

        places = list(np.unique(df_all['county'].values))

        def replace(x):
            if x in places:
                return x
            
            else:
                return 'Other'
            
        df_all['place'] = df_all['place'].apply(lambda x: replace(x))

        le = LabelEncoder()
        df_all['place'] = le.fit_transform(df_all['place'])

        pca = PCA().fit(coordinates)
        df_all['pca_lat'] = pca.transform(coordinates)[:, 0]
        df_all['pca_lon'] = pca.transform(coordinates)[:, 1]

        if use_umap:

            from umap import UMAP

            umap = UMAP(n_components=2,
                    n_neighbors=50,
                    random_state=2023).fit(coordinates)
            df_all['umap_lat'] = umap.transform(coordinates)[:, 0]
            df_all['umap_lon'] = umap.transform(coordinates)[:, 1]

        df_all = rot(df_all)

        return df_all


    elif version=='v2':
        return rot(df_all)