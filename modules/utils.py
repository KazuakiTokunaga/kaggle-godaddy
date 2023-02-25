import numpy as np
import pandas as pd
import pickle

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


def get_df_all(df_train, df_test, categorize=False):

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

    df_all['county'] = (df_all['county'] + df_all['state']).factorize()[0]
    df_all['state_i'] = df_all['state'].factorize()[0]

    if categorize:
        cat_cols = ['county', 'state']
        df_all[cat_cols] = df_all[cat_cols].astype('category')
    
    return df_all

def load_dataset(BASE = '../input/'):

    df_train = pd.read_csv(BASE + 'train.csv',  index_col='row_id')
    df_test = pd.read_csv(BASE + 'test.csv',  index_col='row_id')
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


def smooth_outlier(df_all, max_scale=40):
    print(f'smooth_outlier: max_scale={max_scale}')
    
    outliers = []
    cnt = 0

    for o in df_all.cfips.unique():
        indices = (df_all['cfips']==o)
        tmp = df_all.loc[indices].copy().reset_index(drop=True)
        var = tmp.microbusiness_density.values.copy()
        
        for i in range(max_scale, 0, -1):
            thr = 0.20*np.mean(var[:i])
            difa = abs(var[i]-var[i-1])
            if (difa>=thr):
                var[:i] *= (var[i]/var[i-1])
                outliers.append(o)
                cnt+=1
        var[0] = var[1]*0.99
        df_all.loc[indices, mbd] = var
    
    outliers = np.unique(outliers)
    print(f'# of fixed cfips: {len(outliers)}')
    print(f'# of fixed value: {cnt}')
    
    return df_all

def merge_dataset(df_train, df_test, BASE='../input/', pop=False, census=True, 
                unemploy=True, outlier=False, coord=True, fix_pop=True, categorize=False):

    df_all = get_df_all(df_train, df_test, categorize=categorize)

    if pop:
        df_all = merge_pop(df_all, BASE)
    if census:
        df_all = merge_census_starter(df_all, BASE)
    if unemploy:
        df_all = merge_unemploy(df_all, BASE)
    if coord:
        df_all = merge_coord(df_all, BASE)

    df_census = load_census(BASE)
    if fix_pop:
        df_all = fix_population(df_all, df_census)

    if outlier:
        df_all = smooth_outlier(df_all)
    
    return df_all, df_census


def fix_population(df_all, df_census):

    df_all = df_all.reset_index()
    df_all = df_all.merge(df_census, how='left', on='cfips')

    for year in [2019, 2020, 2021]:
        indices = (df_all['year']==year)
        target_year_str = str(year - 2)
        df_all.loc[indices, mbd] = df_all.loc[indices, mbd] *  df_all.loc[indices, 'adult_2020'] / df_all.loc[indices, f'adult_{target_year_str}']
    
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


def regularize(x, v):
    if x >= 1:
        if x * (1-v) >= 1:
            x *= (1-v)
        else:
            x = None
    else:
        if x * (1+v) <= 1:
            x *= (1+v)
        else:
            x = None
    return x


def insert_trend(df_sub_base, df_all, trend_dict, month_str='_2023-01-01', method='replace'):

    for cfip in trend_dict:
        row_id = str(cfip) + month_str
    
        if method=='replace':
            df_sub_base.loc[row_id, :] = (trend_dict[cfip] * df_all.loc[(df_all['scale']==40)&(df_all['cfips']==cfip), mbd]).values[0]
        elif method=='mean':
            trend_values = (trend_dict[cfip] * df_all.loc[(df_all['scale']==38)&(df_all['cfips']==cfip), mbd]).values[0]
            df_sub_base.loc[row_id, :] = (df_sub_base.loc[row_id].values[0] + trend_values) / 2
        else:
            raise Exception()
        
    return df_sub_base


def adjust_population(df_submission, df_census):

    df_submission = df_submission.reset_index()
    df_submission['cfips'] = df_submission['row_id'].apply(lambda x: int(x.split('_')[0]))
    adult2020 = df_census.set_index('cfips')['adult_2020'].to_dict()
    adult2021 = df_census.set_index('cfips')['adult_2021'].to_dict()
    df_submission['adult2020'] = df_submission['cfips'].map(adult2020)
    df_submission['adult2021'] = df_submission['cfips'].map(adult2021)
    df_submission[mbd] = df_submission[mbd] * df_submission['adult2020'] / df_submission['adult2021']
    df_submission = df_submission.drop(['adult2020','adult2021','cfips'],axis=1)
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
