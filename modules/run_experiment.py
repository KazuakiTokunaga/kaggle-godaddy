import numpy as np
import pandas as pd
import sys

from modules import utils
from modules import models
from modules import preprocess

def main(BASE, external, params):
    mbd = 'microbusiness_density'

    df_train, df_test, df_subm = utils.load_dataset(BASE)
    df_all = utils.merge_dataset(df_train, df_test, external, pop=False, unemploy=False, census=True, coord=True, outlier=True)
    df_census = utils.load_census(external)
    df_all = utils.fix_population(df_all, df_census)

    myinstance = models.LgbmBaseline('kaggle_notebook_test', df_subm, df_all, df_census, start_all_dict=40, save_path=False, params=params)
    myinstance.create_submission(accum=True, target_scale=[41])

