import numpy as np
import pandas as pd
import sys

from modules import utils
from modules import models
from modules import preprocess

def main(BASE, external, params):
    mbd = 'microbusiness_density'

    df_train, df_test, df_subm = utils.load_dataset(BASE)
    df_all = utils.merge_dataset(df_train, df_test, external, unemploy=True, census=True, coord=True, outlier=True)
    df_census = utils.load_census(external)
    df_all = utils.fix_population(df_all, df_census)

    instance_validation = models.LgbmBaseline('validation', df_subm, df_all, df_census, start_all_dict=32, save_path=False, params=params)
    instance_validation.accum_validation(m_len=5)

    instalce_prediction = models.LgbmBaseline('submission', df_subm, df_all, df_census, start_all_dict=40, save_path=False, params=params)
    instalce_prediction.create_submission(accum=True)

