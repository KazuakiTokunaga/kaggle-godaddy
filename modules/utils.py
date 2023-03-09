import numpy as np
import pandas as pd
import pickle

from modules import preprocess


mbd = 'microbusiness_density'

def smape(y_true, y_pred):
    smap = smape_arr(y_true, y_pred)
    
    return np.mean(smap)

def smape_arr(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = (num[pos_ind] / dem[pos_ind]) * 100

    return smap


def get_df_all(df_train, df_test, categorize=False, county=True):

    state_dict = df_train[['cfips', 'state', 'county']]
    state_dict = state_dict.set_index('cfips')
    state_dict = state_dict.drop_duplicates()
    state_dict = state_dict.to_dict()

    df_test['state'] = df_test['cfips'].map(state_dict['state'])
    df_test['county'] = df_test['cfips'].map(state_dict['county'])

    df_all = pd.concat([df_train, df_test], axis=0)

    date_col = 'first_day_of_month'
    df_all[date_col] = pd.to_datetime(df_all[date_col])

    df_all['year'] = df_all[date_col].dt.year
    df_all['month'] = df_all[date_col].dt.month
    df_all['scale'] = (df_all[date_col] - df_all[date_col].min()).dt.days
    df_all['scale'] = df_all['scale'].factorize()[0]

    df_all = df_all.drop(columns='first_day_of_month')
    df_all.sort_index(inplace=True)

    if county:
        df_all['county'] = (df_all['county'] + df_all['state']).factorize()[0]
    
    df_all['state_i'] = df_all['state'].factorize()[0]

    if categorize:
        cat_cols = ['county', 'state']
        df_all[cat_cols] = df_all[cat_cols].astype('category')
    
    return df_all

def load_dataset(BASE = '../input/', subm=''):

    df_train = pd.read_csv(BASE + 'train.csv',  index_col='row_id')
    df_test = pd.read_csv(BASE + 'test.csv',  index_col='row_id')
    
    if subm:
        df_subm = pd.read_csv(subm,  index_col='row_id')
    else:
        df_subm = pd.read_csv(BASE + 'sample_submission.csv',  index_col='row_id')

    df_revealed_test = pd.read_csv(BASE + 'revealed_test.csv', index_col='row_id')

    df_train = df_train[~(df_train['first_day_of_month']>='2022-11-01')]
    df_test = df_test[df_test['first_day_of_month']>='2023-01-01']

    df_train = pd.concat([df_train, df_revealed_test])

    return df_train, df_test, df_subm

def load_census(BASE = '../input/'):

    COLS = ['GEO_ID','S0101_C01_026E']

    for i in [2017, 2018, 2019, 2020, 2021]:
        df_add = pd.read_csv(BASE + f'census-data-for-godaddy/ACSST5Y{i}.S0101-Data.csv',usecols=COLS)
        df_add = df_add.iloc[1:] 
        df_add['cfips'] = df_add.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
        df_add['S0101_C01_026E'] = df_add['S0101_C01_026E'].astype('int')
        df_add.rename(columns={'S0101_C01_026E': f'adult_{i}'}, inplace=True)
        df_add.drop(['GEO_ID'], inplace=True, axis=1)
        
        if i == 2017:
            df_pop = df_add.copy()
        else:
            df_pop = df_pop.merge(df_add, how='left', on='cfips')

    return df_pop    


def merge_census_starter(df_all, BASE='../input/'):
    
    df_census = pd.read_csv(BASE + 'census_starter.csv', index_col='cfips')

    df_all = df_all.reset_index()
    df_all = df_all.set_index('cfips')

    df_all[df_census.columns] = df_census

    df_all = df_all.reset_index()
    df_all = df_all.set_index('row_id')

    return df_all


def merge_pop(df_all, BASE='../input/'):
    
    df_pop = pd.read_csv(BASE + 'PopulationEstimates.csv')

    p1990 = 'Population 1990'
    p2000 = 'Population 2000'
    p2010 = 'Population 2010'
    p2020 = 'Population 2020'
    p2021 = 'Population 2021'

    df_pop.loc[df_pop[p2000].isna(), p2000] = df_pop.loc[df_pop[p2000].isna(), p1990]
    df_pop.loc[df_pop[p2010].isna(), p2010] = df_pop.loc[df_pop[p2010].isna(), p2000]
    df_pop.loc[df_pop[p2020].isna(), p2020] = df_pop.loc[df_pop[p2020].isna(), p2000]
    df_pop.loc[df_pop[p2021].isna(), p2021] = df_pop.loc[df_pop[p2021].isna(), p2020]
    df_pop.loc[df_pop[p2010].isna(), p2010] = df_pop.loc[df_pop[p2010].isna(), p2020]
    df_pop.loc[df_pop[p2000].isna(), p2000] = df_pop.loc[df_pop[p2000].isna(), p2010]

    df_pop['pop_rate1'] = df_pop[p2010] / df_pop[p2000]
    df_pop['pop_rate2'] = df_pop[p2020] / df_pop[p2010]
    df_pop['pop_rate3'] = df_pop[p2021] / df_pop[p2020]
    df_pop.rename(columns={p2020: 'p2020', p2021:'p2021'}, inplace=True)

    df_all_t = df_all.reset_index()
    df_all_t = df_all_t.merge(df_pop[['cfips', 'p2020', 'p2021', 'pop_rate1', 'pop_rate2', 'pop_rate3']], how='left', on='cfips').set_index('row_id')

    return df_all_t

def merge_unemploy(df_all, BASE='../input/'):

    df_unemploy = pd.read_csv(BASE + 'Unemployment.csv', index_col='cfips')

    rate = 'Unemployment_rate'
    rate2019 = f'{rate}_2019'
    rate2020 = f'{rate}_2020'
    rate2021 = f'{rate}_2021'

    cs = [rate2019, rate2020, rate2021]

    df_cs = df_unemploy[cs].copy()
    mean2019 = df_cs[rate2019].mean()
    mean2020 = df_cs[rate2020].mean()
    mean2021 = df_cs[rate2021].mean()
    df_cs.loc[15005] = [mean2019,mean2020,mean2021]

    df_cs.loc[df_cs[rate2019].isna(), rate2019] = mean2019

    idx = df_cs[rate2021].isna()
    df_cs.loc[idx, rate2021] = df_cs.loc[idx, rate2019]

    idx = df_cs[rate2020].isna()
    df_cs.loc[idx, rate2020] = (df_cs.loc[idx, rate2019] + df_cs.loc[idx, rate2021]) / 2

    df_cs['unemploy_uprate1'] = df_cs[rate2020] / df_cs[rate2019]
    df_cs['unemploy_uprate2'] = df_cs[rate2021] / df_cs[rate2020]
    df_cs['unemploy_uprate3'] = df_cs[rate2021] / df_cs[rate2019]

    df_all = df_all.reset_index()
    df_all = df_all.merge(df_cs.reset_index(), how='left', on='cfips').set_index('row_id')

    return df_all

def merge_coord(df_all, BASE='../input/'):

    df_coords = pd.read_csv(BASE + "cfips_location.csv")
    df_all = df_all.reset_index()
    df_all = df_all.merge(df_coords.drop("name", axis=1), on="cfips")
    df_all = df_all.set_index('row_id')

    return df_all

def smooth_outlier(df_all_base, max_scale=40, method='v1'):
    print(f'smooth_outlier: max_scale={max_scale}')
    
    outliers = []
    cnt = 0

    df_all = df_all_base.copy()
    if method=='v1':
        
        for o in df_all.cfips.unique():
            indices = (df_all['cfips']==o)
            tmp = df_all.loc[indices].copy().reset_index(drop=True)
            var = tmp.microbusiness_density.values.copy()
            
            if o not in [28055, 48301]:
                for i in range(max_scale, 0, -1):
                    thr = 0.20*np.mean(var[:i])
                    difa = abs(var[i]-var[i-1])
                    if (difa>=thr):
                        var[:i] *= (var[i]/var[i-1])
                        outliers.append(o)
                        cnt+=1
            var[0] = var[1]*0.99
            df_all.loc[indices, mbd] = var

    elif method=='v2':

        for o in df_all.cfips.unique(): 
            indices = (df_all['cfips'] == o)  
            tmp = df_all.loc[indices].copy().reset_index(drop=True)  
            var = tmp.microbusiness_density.values.copy()
            
            if o not in [28055, 48301]:
                for i in range(max_scale-3, 2, -1):
                    thr = 0.10 * np.mean(var[:i]) 
                    difa = var[i] - var[i - 1] 
                    if (difa >= thr) or (difa <= -thr):  
                        if difa > 0:
                            var[:i] += difa - 0.0045 
                        else:
                            var[:i] += difa + 0.0043 
                        outliers.append(o)
                        cnt+=1
            var[0] = var[1] * 0.99
            df_all.loc[indices, mbd] = var

    elif method=='v3':

        for o in df_all.cfips.unique(): 
            indices = (df_all['cfips'] == o)  
            tmp = df_all.loc[indices].copy().reset_index(drop=True)  
            var = tmp.microbusiness_density.values.copy()
            
            if o not in [28055, 48301]:
                for i in range(max_scale-3, 2, -1):
                    thr = 0.10 * np.mean(var[:i]) 
                    difa = var[i] - var[i - 1] 
                    if (difa >= thr) or (difa <= -thr):  
                        if difa > 0:
                            var[:i] += difa - 0.003
                        else:
                            var[:i] += difa + 0.003 
                        outliers.append(o)
                        cnt+=1
            var[0] = var[1] * 0.99
            df_all.loc[indices, mbd] = var

    else:
        print('No smooth.')

    outliers = np.unique(outliers)
    print(f'used method: {method}')
    print(f'# of fixed cfips: {len(outliers)}')
    print(f'# of fixed value: {cnt}')
    
    return df_all

def merge_coest(df_all,  BASE='../input/'):

    df_co_est = pd.read_csv(BASE + "co-est2021-alldata.csv", encoding='latin-1')
    df_co_est["cfips"] = df_co_est.STATE*1000 + df_co_est.COUNTY
    co_columns = [
        'cfips',
        'SUMLEV',
        'DIVISION',
        'ESTIMATESBASE2020',
        'POPESTIMATE2020',
        'POPESTIMATE2021',
        'NPOPCHG2020',
        'NPOPCHG2021',
        'BIRTHS2020',
        'BIRTHS2021',
        'DEATHS2020',
        'DEATHS2021',
        'NATURALCHG2020',
        'NATURALCHG2021',
        'INTERNATIONALMIG2020',
        'INTERNATIONALMIG2021',
        'DOMESTICMIG2020',
        'DOMESTICMIG2021',
        'NETMIG2020',
        'NETMIG2021',
        'RESIDUAL2020',
        'RESIDUAL2021',
        'GQESTIMATESBASE2020',
        'GQESTIMATES2020',
        'GQESTIMATES2021',
        'RBIRTH2021',
        'RDEATH2021',
        'RNATURALCHG2021',
        'RINTERNATIONALMIG2021',
        'RDOMESTICMIG2021',
        'RNETMIG2021'
    ]
    df_all = df_all.reset_index()
    df_all = df_all.merge(df_co_est[co_columns], on="cfips")
    df_all = df_all.set_index('row_id')

    return df_all


def merge_dataset(df_train, df_test, BASE='../input/', pop=False, census=True, county=True,
                unemploy=True, coord=True, co_est=True, fix_pop=True, 
                add_location=False, use_umap=False, categorize=False, merge41=False, df_subm='', mbd_origin='after'):

    df_all = get_df_all(df_train, df_test, categorize=categorize, county=county)

    if mbd_origin=='before':
        df_all['mbd_origin'] = df_all[mbd]

    if pop:
        df_all = merge_pop(df_all, BASE)
    if census:
        df_all = merge_census_starter(df_all, BASE)
    if unemploy:
        df_all = merge_unemploy(df_all, BASE)
    if coord:
        df_all = merge_coord(df_all, BASE)
    if co_est:
        df_all = merge_coest(df_all, BASE)

    df_census = load_census(BASE)
    if fix_pop:
        df_all = fix_population(df_all, df_census)
    
    if add_location:
        df_all = preprocess.add_location(df_all, use_umap)

    if merge41:
        df_all = merge_scale41(df_all, df_subm, df_census)

    if mbd_origin=='after':
        df_all['mbd_origin'] = df_all[mbd]    

    return df_all, df_census


def fix_population(df_all, df_census):

    df_all = df_all.reset_index()
    df_all = df_all.merge(df_census, how='left', on='cfips')

    for year in [2019, 2020, 2021]:
        indices = (df_all['year']==year)
        target_year_str = str(year - 2)
        df_all.loc[indices, mbd] = np.round(100 * df_all.loc[indices, 'active'] /  df_all.loc[indices, 'adult_2020'], 6)
    
    drop_columns = list(df_census.columns)
    drop_columns.remove('cfips')
    df_all = df_all.drop(drop_columns, axis=1)
    df_all = df_all.set_index('row_id')

    return df_all


def save_pickle(obj, filename):
    save_path = f'../output/{filename}.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    save_path = f'../output/{filename}.pickle'
    with open(save_path, 'rb') as f:
        res = pickle.load(f)

    return res


def insert_trend(df_submission, df_all, df_census, trend_dict, fix_pop=True, method='replace'):

    df_submission = df_submission.reset_index()
    df_submission['cfips'] = df_submission['row_id'].apply(lambda x: int(x.split('_')[0]))
    df_submission['month'] = df_submission['row_id'].apply(lambda x: x.split('_')[1])

    target = 41
    train = target - 1

    df_extract = df_all[df_all['scale']==train].copy()
    df_extract['mbd_trend'] = df_extract['mbd_origin'] * df_extract['cfips'].map(trend_dict)
    if fix_pop:
        adult2020 = df_census.set_index('cfips')['adult_2020'].to_dict()
        adult2021 = df_census.set_index('cfips')['adult_2021'].to_dict()
        df_extract['adult2020'] = df_extract['cfips'].map(adult2020)
        df_extract['adult2021'] = df_extract['cfips'].map(adult2021)
        df_extract['mbd_trend'] = df_extract['mbd_trend'] * df_extract['adult2020'] / df_extract['adult2021']

    var_dict = df_extract[~df_extract['mbd_trend'].isna()].reset_index()[['cfips', 'mbd_trend']].set_index('cfips').to_dict()['mbd_trend']
    df_submission['trend'] = df_submission['cfips'].map(var_dict)
    idx = (~df_submission['trend'].isna())&(df_submission['month']=='2023-01-01')
    
    if method=='replace':
        df_submission.loc[idx, mbd] = df_submission.loc[idx, 'trend']
    elif method=='mean':
        df_submission.loc[idx, mbd] = (df_submission.loc[idx, 'trend'] + df_submission.loc[idx, mbd]) / 2
    else:
        raise Exception('Wrong Method.')

    df_submission = df_submission.drop(['trend', 'cfips', 'month'], axis=1).set_index('row_id')
        
    return df_submission, df_extract, var_dict


def adjust_population(df_submission, df_census, start_month='2023-01-01'):

    df_submission = df_submission.reset_index()
    df_submission['cfips'] = df_submission['row_id'].apply(lambda x: int(x.split('_')[0]))
    df_submission['month'] = df_submission['row_id'].apply(lambda x: x.split('_')[1])
    adult2020 = df_census.set_index('cfips')['adult_2020'].to_dict()
    adult2021 = df_census.set_index('cfips')['adult_2021'].to_dict()
    df_submission['adult2020'] = df_submission['cfips'].map(adult2020)
    df_submission['adult2021'] = df_submission['cfips'].map(adult2021)

    idx = (df_submission['month']>=start_month)
    df_submission.loc[idx, mbd] = df_submission.loc[idx, mbd] * df_submission.loc[idx, 'adult2020'] / df_submission.loc[idx, 'adult2021']
    df_submission = df_submission.drop(['adult2020','adult2021','cfips', 'month'],axis=1)
    df_submission = df_submission.set_index('row_id')

    return df_submission


def compare_submission(df_submission, filename):

    df_submission = df_submission.reset_index()
    df_submission['cfips'] = df_submission['row_id'].apply(lambda x: int(x.split('_')[0]))
    df_submission['month'] = df_submission['row_id'].apply(lambda x: x.split('_')[1])
    df_subm202301 = df_submission[df_submission['month']=='2023-01-01']

    df_baseline = pd.read_csv(f'../submission/{filename}.csv')
    df_baseline['cfips'] = df_baseline['row_id'].apply(lambda x: int(x.split('_')[0]))
    df_baseline['month'] = df_baseline['row_id'].apply(lambda x: x.split('_')[1])
    df_baseline202301 = df_baseline[df_baseline['month']=='2023-01-01'].copy()
    df_baseline202301.rename(columns={mbd: 'baseline'}, inplace=True)

    df_merged = df_subm202301.merge(df_baseline202301[['row_id', 'baseline']], how='left', on='row_id')
    df_merged['smape'] = smape_arr(df_merged[mbd], df_merged['baseline'])

    return df_merged


def merge_scale41(df_all, df_submission, df_census):
    print('merge scale=41 of df_submission to df_all.')
    
    df_all = df_all.reset_index()
    
    df_subt = df_submission.copy().reset_index().rename(columns={mbd: 'mbd_pred'})
    df_subt['month'] = df_subt['row_id'].apply(lambda x: x.split('_')[1])
    df_subt = df_subt[df_subt['month']=='2023-01-01']    
    df_all = df_all.merge(df_subt.drop(['month'], axis=1), how='left', on='row_id')
    
    idx = ~df_all['mbd_pred'].isna()
    df_all.loc[idx, mbd] = df_all.loc[idx, 'mbd_pred']
    
    adult2020 = df_census.set_index('cfips')['adult_2020'].to_dict()
    adult2021 = df_census.set_index('cfips')['adult_2021'].to_dict()
    df_all['adult2020'] = df_all['cfips'].map(adult2020)
    df_all['adult2021'] = df_all['cfips'].map(adult2021)
    df_all.loc[idx, 'active'] = np.round(df_all.loc[idx, 'mbd_pred'] * df_all.loc[idx, 'adult2021'] / 100)
    df_all.loc[idx, mbd] = np.round(100 * df_all.loc[idx, 'active'] / df_all.loc[idx, 'adult2020'], 6)

    df_all = df_all.drop(['mbd_pred', 'adult2020', 'adult2021'], axis=1).set_index('row_id')

    return df_all

def create_df_season(df_all, active_thre=5000, validate=True, abs_thre=[-0.006, 0.006], v_clip=[-0.01, 0.01]):

    max_scale = 32 if validate else 40
    print(f'create df_season, max_scale: {max_scale}, validate: {validate}')

    df_t = df_all[(df_all['scale']<=max_scale)&(df_all[f'select_lastactive{max_scale}']>=active_thre)].copy()
    df_t['select_rate1'] = df_t['select_rate1'].clip(v_clip[0], v_clip[1])
    df_t['cnt'] = df_t['select_rate1'].apply(lambda x: 1 if x>0 else -1)

    df_s = df_t.groupby(['cfips', 'month']).agg(['sum', 'count'])[['select_rate1', 'cnt']].reset_index()
    df_s.columns = ['cfips', 'month'] + ['_'.join(col) for col in df_s.columns.values[2:]]
    df_s['select_rate1_mean'] = df_s['select_rate1_sum'] / df_s['cnt_count']

    if validate:
        df_s['scale'] = df_s['month'] + 28
    else:
        df_s['scale'] = df_s['month'] + 40

    up_idx = (df_s['select_rate1_mean']>=abs_thre[1])&(df_s['cnt_sum']==df_s['cnt_count'])
    df_up = df_s.loc[up_idx, ['cfips', 'select_rate1_mean', 'scale']]

    down_idx = (df_s['select_rate1_mean']<=-1*abs_thre[0])&(-1*df_s['cnt_sum']==df_s['cnt_count'])
    df_down = df_s.loc[down_idx, ['cfips', 'select_rate1_mean', 'scale']]
    

    return pd.concat([df_up, df_down])


def round_integer(df_submission, df_census):
        
    df_submission = df_submission.reset_index()
    df_submission['cfips'] = df_submission['row_id'].apply(lambda x: int(x.split('_')[0]))
    df_submission['month'] = df_submission['row_id'].apply(lambda x: x.split('_')[1])
    adult2021 = df_census.set_index('cfips')['adult_2021'].to_dict()
    df_submission['adult2021'] = df_submission['cfips'].map(adult2021)

    idx = (df_submission['month']>='2023-01-01')
    df_submission.loc[idx, mbd] = np.round(df_submission.loc[idx, mbd] * df_submission.loc[idx, 'adult2021'] / 100) / df_submission.loc[idx, 'adult2021'] * 100
    df_submission = df_submission.drop(['adult2021','cfips', 'month'],axis=1)
    df_submission = df_submission.set_index('row_id')

    return df_submission