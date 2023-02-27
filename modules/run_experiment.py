import numpy as np
import pandas as pd
import sys

from modules import utils
from modules import models
from modules import preprocess

def main(BASE, external, params, fix_pop=True, use_umap=False, add_location=True):
    mbd = 'microbusiness_density'

    df_train, df_test, df_subm = utils.load_dataset(BASE)
    df_all, df_census = utils.merge_dataset(df_train, df_test, BASE=external, unemploy=False, census=True, fix_pop=fix_pop, coord=True, add_location=add_location, use_umap=use_umap, outlier=True)

    instance_validation = models.LgbmBaseline('validation', df_subm, df_all, df_census, start_all_dict=32, save_path=False, params=params)
    instance_validation.accum_validation()

    instalce_prediction = models.LgbmBaseline('submission', df_subm, df_all, df_census, start_all_dict=40, save_path=False, params=params)
    instalce_prediction.create_submission()
