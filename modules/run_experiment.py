import numpy as np
import pandas as pd
import sys

from modules import utils
from modules import models
from modules import preprocess

def main(BASE, external, params, trend_params, validate=True, add_location=True, coord=True,
        fix_pop=True, use_umap=False, co_est=True, outlier=True, outlier_method='v1', census=True,
        merge41=False, subm='/kaggle/input/godaddymy/submission_13769_trend.csv'):

    mbd = 'microbusiness_density'

    df_train, df_test, df_subm = utils.load_dataset(BASE, subm)
    df_all, df_census = utils.merge_dataset(df_train, df_test, 
        BASE=external, 
        unemploy=False, 
        census=census,  
        fix_pop=fix_pop, 
        coord=coord, 
        co_est=co_est,
        add_location=add_location, 
        use_umap=use_umap, 
        outlier=outlier,
        outlier_method=outlier_method,
        merge41=merge41,
        df_subm=df_subm
    )

    if validate:
        instance_validation = models.LgbmBaseline('validation', df_subm, df_all, df_census, start_all_dict=32, save_path=False, params=params, trend_params=trend_params)
        instance_validation.accum_validation()

    if merge41:
        instalce_prediction = models.LgbmBaseline('submission', df_subm, df_all, df_census, start_all_dict=40, save_path=False, params=params, trend_params=trend_params)
        instalce_prediction.create_submission(target_scale=[42,43,44,45])
    else:
        instalce_prediction = models.LgbmBaseline('submission', df_subm, df_all, df_census, start_all_dict=40, save_path=False, params=params, trend_params=trend_params)
        instalce_prediction.create_submission()
