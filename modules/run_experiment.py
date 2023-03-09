import numpy as np
import pandas as pd
import sys

from modules import utils
from modules import models
from modules import preprocess

def main(BASE, external, params, trend_params, season_params, validate=True, add_location=True, coord=True,
        fix_pop=True, use_umap=False, co_est=True, census=True, unemploy=False,
        subm='/kaggle/input/godaddymy/submission_13732_trend.csv'):

    mbd = 'microbusiness_density'
    merge41 = True if params.get('start_max_scale')==41 else False

    df_train, df_test, df_subm = utils.load_dataset(BASE, subm)
    df_all, df_census = utils.merge_dataset(df_train, df_test, 
        BASE=external, 
        unemploy=unemploy, 
        census=census,  
        fix_pop=fix_pop, 
        coord=coord, 
        co_est=co_est,
        add_location=add_location, 
        use_umap=use_umap, 
        merge41=merge41,
        df_subm=df_subm
    )

    if validate:
        instance_validation = models.LgbmBaseline('validation', df_subm, df_all, df_census, save_path=False, params=params, trend_params=trend_params, season_params=season_params)
        instance_validation.accum_validation()

    params['start_all_dict'] = 40
    if merge41:
        instalce_prediction = models.LgbmBaseline('submission', df_subm, df_all, df_census, save_path=False, params=params, trend_params=trend_params, season_params=season_params)
        instalce_prediction.create_submission(target_scale=[42,43,44,45])
    else:
        instalce_prediction = models.LgbmBaseline('submission', df_subm, df_all, df_census, save_path=False, params=params, trend_params=trend_params, season_params=season_params)
        instalce_prediction.create_submission()
