import numpy as np
import pandas as pd
import sys

from modules import utils
from modules import models
from modules import preprocess

def main(BASE, external, params, trend_params, fix_pop=True, use_umap=False, co_est=True, outlier_method='v1'):
    mbd = 'microbusiness_density'

    df_train, df_test, df_subm = utils.load_dataset(BASE)
    df_all, df_census = utils.merge_dataset(df_train, df_test, 
        BASE=external, 
        unemploy=False, 
        census=True, 
        fix_pop=fix_pop, 
        coord=True, 
        co_est=co_est,
        add_location=True, 
        use_umap=use_umap, 
        outlier=True,
        outlier_method=outlier_method
    )

    instance_validation = models.LgbmBaseline('validation', df_subm, df_all, df_census, start_all_dict=32, save_path=False, params=params, trend_params=trend_params)
    instance_validation.accum_validation()

    instalce_prediction = models.LgbmBaseline('submission', df_subm, df_all, df_census, start_all_dict=40, save_path=False, params=params, trend_params=trend_params)
    instalce_prediction.create_submission()
