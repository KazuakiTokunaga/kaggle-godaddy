import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from modules import utils

# これはリークする
def add_seasonal_effect(df_all, max_scale=38):

    idx = (df_all['scale']>=4)&(df_all['scale']<=max_scale)
    df_cfips_month = df_all.loc[idx].groupby(['cfips', 'month']).sum()[['select_rate1', 'select_rate2', 'select_rate3']].reset_index()

    rename_dict = dict()
    for i in range(1, 4):
        rename_dict[f'select_rate{i}'] = f'cfips_sum_rate{i}'
    df_cfips_month.rename(columns=rename_dict, inplace=True)
                                                                    
    df_all = df_all.reset_index()
    df_all = df_all.merge(df_cfips_month, how='left', on=['cfips', 'month']).set_index('row_id')

    return df_all