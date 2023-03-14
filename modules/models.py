import datetime
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from modules import utils
from modules import preprocess
from sklearn.pipeline import Pipeline

mbd = 'microbusiness_density'


def get_model(algo='lgb', light=False, lgbm_params = {
    'n_iter': 30,
    'learning_rate': 0.0036,
    'colsample_bytree': 0.884,
    'colsample_bynode': 0.101,
    'max_depth': 8,
    'lambda_l2': 0.5,
    'num_leaves': 61,
    'min_data_in_leaf': 213
}):

    params = {
        'n_iter': lgbm_params['n_iter'],
        'verbosity': -1,
        'objective': 'l1',
        'random_state': 42,
        'colsample_bytree': lgbm_params['colsample_bytree'],
        'colsample_bynode': lgbm_params['colsample_bynode'],
        'max_depth': lgbm_params['max_depth'],
        'learning_rate': lgbm_params['learning_rate'],
        'lambda_l2': lgbm_params['lambda_l2'],
        'num_leaves': lgbm_params['num_leaves'],
        "seed": 42,
        'min_data_in_leaf': lgbm_params['min_data_in_leaf']
    }

    if light:
        params['n_iter']=30

    lgb_model = lgb.LGBMRegressor(**params)
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        tree_method="hist",
        n_estimators=795,
        learning_rate=0.0075,
        max_leaves = 17,
        subsample=0.50,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
    )
    
    cat_model = cat.CatBoostRegressor(
        iterations=800,
        loss_function="MAPE",
        verbose=0,
        grow_policy='SymmetricTree',
        learning_rate=0.035,
        max_depth=6,
        l2_leaf_reg=0.2,
        subsample=0.50,
        max_bin=4096,
    )

    cat_model2 = cat.CatBoostRegressor(
        iterations=2000,
        loss_function="MAPE",
        verbose=0,
        grow_policy='SymmetricTree',
        learning_rate=0.035,
        colsample_bylevel=0.8,
        max_depth=5,
        l2_leaf_reg=0.2,
        subsample=0.70,
        max_bin=4096,
    )

    if algo=='lgb':
        
        print(f'use lgb')
        return lgb_model
    
    elif algo=='xgb':

        print('use xgb.')
        return xgb_model
    
    elif algo=='ensemble':

        print('use ensemble.')
        return VotingRegressor([
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cat', cat_model)],weights=[3,1,3]
        )

    elif algo=='tuned_ensemble':

        print('use tuned_ensemble.')
        return VotingRegressor([
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cat', cat_model2)],weights=[3,1,3]
        )
    
    elif algo=='cat ensemble':

        return VotingRegressor([
            ('cat2', cat_model2),
            ('cat', cat_model)],weights=[5,3]
        )
    
    elif algo=='cat only':

        print('use tuned_catboost')
        return cat_model2


class LgbmBaseline():

    def __init__(self, run_fold_name, df_subm, df_all, df_census, save_path=True, params={
        "act_thre": 2.00,
        "abs_thre": 1.00,
        "USE_LAG": 5,
        "USE_TREND": False,
        "USE_SHORT_TREND": False,
        "blacklist": [],
        "blacklistcfips": [],
        "clip": (None, None),
        "model": 'lgbm',
        "light": False,
        "lgbm_params": {
            'n_iter': 30,
            'learning_rate': 0.0036,
            'colsample_bytree': 0.884,
            'colsample_bynode': 0.101,
            'max_depth': 8,
            'lambda_l2': 0.5,
            'num_leaves': 61,
            'min_data_in_leaf': 213
        },
        "max_window": 12,
        "start_max_scale": 40,
        "start_all_dict": 32,
        "smooth_method": 'v3',
        "v3_thre": 0.1,
        "v3_adjust": 0.003,
        "save_output_dic": False,
        "USE_SEASON": False
    }, trend_params = {
        "high_trend_params": {
            1: {
                'params':{
                    'n':3,
                    'thre':3,
                    'thre_r':0,
                    'lower_bound': 15000,
                    'upper_bound': 999999,
                    'use_regularize': True,
                    'v_regularize': [0.01, 0.008],
                    'v_clip':[0.999, 1.004]
                },
                'method': 'mean'
            },
            2: {
                'params':{
                    'n':4,
                    'thre':4,
                    'thre_r':0,
                    'lower_bound': 15000,
                    'upper_bound': 999999,
                    'use_regularize': True,
                    'v_regularize': [0.01, 0.008],
                    'v_clip':[0.999, 1.004]
                },
                'method': 'mean'
            },
            3: {
                'params':{
                    'n':5,
                    'thre':5,
                    'thre_r':0,
                    'lower_bound': 15000,
                    'upper_bound': 999999,
                    'use_regularize': True,
                    'v_regularize': [0.01, 0.008],
                    'v_clip':[0.999, 1.004]
                },
                'method': 'mean'
            }
        },
        "low_trend_params": {
            1: {
                'params':{
                    'n':3,
                    'thre':3,
                    'thre_r':0,
                    'lower_bound': 60,
                    'upper_bound': 140,
                    'use_regularize': True,
                    'v_regularize': [0.03, 0.02],
                    'v_clip':[0.999, 1.004]
                },
                'method': 'replace'
            },
            2: {
                'params':{
                    'n':4,
                    'thre':4,
                    'thre_r':0,
                    'lower_bound': 60,
                    'upper_bound': 140,
                    'use_regularize': True,
                    'v_regularize': [0.03, 0.02],
                    'v_clip':[0.999, 1.004]
                },
                'method': 'replace'
            },
            3: {
                'params':{
                    'n':5,
                    'thre':5,
                    'thre_r':0,
                    'lower_bound': 60,
                    'upper_bound': 140,
                    'use_regularize': True,
                    'v_regularize': [0.03, 0.02],
                    'v_clip':[0.999, 1.004]
                },
                'method': 'replace'
            }
        },
    }, season_params = {
        "abs_thre": [-0.003, 0.006],
        "active_thre": 5000,
        "v_clip": [-0.01, 0.01],
        "method": "trend_mean"
    }):

        self.run_fold_name = run_fold_name
        self.df_subm = df_subm
        self.df_census = df_census
        self.output_dic = '../output/'
        self.trend_params = trend_params
        self.season_params = season_params

        self.act_thre = params['act_thre']
        self.abs_thre = params['abs_thre']
        self.USE_LAG = params['USE_LAG']
        self.USE_TREND = params['USE_TREND']
        self.USE_SHORT_TREND = params['USE_SHORT_TREND']
        self.blacklist = params['blacklist']
        self.blacklistcfips = params['blacklistcfips']
        self.clip = params['clip']
        self.model = params['model']
        self.max_window = params['max_window']
        self.start_max_scale = params.get('start_max_scale') if params.get('start_max_scale') else 40
        self.smooth_method = params.get('smooth_method') if params.get('smooth_method') else 'v3'
        self.start_all_dict = params.get('start_all_dict') if params.get('start_all_dict') else 32
        self.save_output_dic = params.get('save_output_dic') if params.get('save_output_dic') else False
        self.USE_SEASON = params.get('USE_SEASON') if params.get('USE_SEASON') else False
        self.v3_adjust = params.get('v3_adjust') if params.get('v3_adjust') else 0.003
        self.v3_thre = params.get('v3_thre') if params.get('v3_thre') else 0.1
        self.lgbm_params = params.get('lgbm_params')

        self.light = params.get('light')
        self.save_path = save_path
        self.print_feature = False
        self.accum_cnt = 0
        self.output_dic = dict()

        self.df_all_dict = dict()
        self.df_all_dict_original = dict()
        for i in range(self.start_all_dict, self.start_max_scale+1):
            print(f'create df_all_dict[{i}] and df_all_dict_original[{i}]')
            self.df_all_dict[i] = preprocess.add_lag_features(
                df_all, 
                max_scale=i, 
                USE_LAG = self.USE_LAG, 
                max_window=self.max_window,
                smooth=True,
                smooth_method=self.smooth_method,
                v3_thre = self.v3_thre,
                v3_adjust = self.v3_adjust
                )
            self.df_all_dict_original[i] = self.df_all_dict[i]

        self.df_all = self.df_all_dict[self.start_max_scale]
    
        self.output_features = ['cfips', 'county', 'state', 'state_i', 'microbusiness_density', 'mbd_origin', 'active', 'year','month', 'scale', 
                                'mbd_pred', 'mbd_model', 'mbd_last', 'mbd_trend', 'mbd_short_trend', 'mbd_season', 'y_base', 'y_pred', 'smape', 'smape_origin']


    def get_df_all_dict(self, train_times, smooth=False):
        
        for i in train_times:
            last_exist_scale = i - self.accum_cnt
            
            print(f'create df_all_dict[{i}] with {self.accum_cnt} predicted months from df_all_dict_original[{last_exist_scale}].')
            output1 = self.output_dic[self.accum_cnt]
            r1 = output1[output1['scale']==i].reset_index()
            for c in range(1, self.accum_cnt):
                output2 = self.output_dic[self.accum_cnt-c]
                r2 = output2[output2['scale']==i-c].reset_index()
                r1 = pd.concat([r1, r2])
            
            df_all_t = self.df_all_dict_original[i - self.accum_cnt]
            df_merged = df_all_t.merge(r1[['row_id', 'mbd_pred']], how='left', on='row_id').set_index('row_id')
            print('insert mbd and mbd_origin of scale:', df_merged.loc[~df_merged['mbd_pred'].isna(), 'scale'].unique())
            df_merged.loc[~df_merged['mbd_pred'].isna(), mbd] = df_merged.loc[~df_merged['mbd_pred'].isna(), 'mbd_pred']
            df_merged.loc[~df_merged['mbd_pred'].isna(), 'mbd_origin'] = df_merged.loc[~df_merged['mbd_pred'].isna(), mbd]
            df_merged.drop(['mbd_pred'], axis=1, inplace=True)
            
            idx = df_merged['scale']>last_exist_scale
            df_merged.loc[idx, 'active'] =  (df_merged.loc[idx, f'select_lastactive{last_exist_scale}'] / df_merged.loc[idx, f'select_lastmbd{last_exist_scale}']) * df_merged.loc[idx, mbd]
            df_merged.loc[df_merged['active'].isna(), 'active'] = 0

            df_all_t = preprocess.add_lag_features(df_merged, 
                max_scale=i, 
                USE_LAG=self.USE_LAG, 
                max_window=self.max_window,
                smooth=smooth, 
                smooth_method=self.smooth_method,
                v3_thre = self.v3_thre,
                v3_adjust = self.v3_adjust
            )
            
            self.df_all_dict[i] = df_all_t


    def run_fit_predict(self, valid_time, pred_m):

        train_times = valid_time - pred_m
        
        print('valid_times: ', valid_time)
        print('pred_m: ', pred_m)
        print('train_times: ', train_times)

        print(f'use df_all_dict[{train_times}]')
        df_all = self.df_all_dict[train_times]

        target = f'select_rate{pred_m}'
        features = preprocess.create_features(df_all, pred_m, train_times, self.USE_LAG)
        if not self.print_feature:
            print(features)
            self.print_feature = True
        
        # Extract Valid and Train data.
        train_indices = (df_all['scale']<=train_times) & (df_all['scale']>=2) & (df_all[f'select_lastactive{train_times}']>self.act_thre) & (df_all[f'select_lastmbd{train_times}']>self.abs_thre)
        X_train = df_all.loc[train_indices, features]
        y_train = df_all.loc[train_indices, target]

        df_valid =  df_all.loc[df_all['scale']==valid_time].copy()
        valid_indices = (df_valid[f'select_lastactive{train_times}']>self.act_thre) & (~df_valid['cfips'].isin(self.blacklistcfips))  & (~df_valid['state'].isin(self.blacklist)) & (df_all[f'select_lastmbd{train_times}']>self.abs_thre)
        X_valid = df_valid.loc[valid_indices, features]
        y_valid = df_valid.loc[valid_indices, target]
        
        # Create Model and predict.
        model = get_model(algo=self.model, light=self.light,lgbm_params=self.lgbm_params)
        model.fit(X_train, y_train.clip(self.clip[0], self.clip[1]))
        y_pred = model.predict(X_valid)
        
        # Use Model result.
        df_valid['y_pred'] = 1
        df_valid.loc[valid_indices, 'y_pred'] = y_pred
        
        # Convert y_pred to microbusiness_density prediction and create output dataset.
        base_indices = (df_all['scale']==train_times)
        base_y = df_all.loc[base_indices, ['cfips', mbd]]
        base_dict = base_y.set_index('cfips').to_dict()
        df_valid['y_base'] = df_valid['cfips'].map(base_dict[mbd])
        df_valid['mbd_pred'] = df_valid['y_base'] * (df_valid['y_pred']+1)
        df_valid.loc[valid_indices, 'mbd_model'] = df_valid['mbd_pred']
        
        # Use Last Value.
        df_valid['mbd_last'] = df_valid[f'select_lastmbd{train_times}']
        lastvalue_indices = ~(valid_indices)
        df_valid.loc[lastvalue_indices, 'mbd_pred'] = df_valid.loc[lastvalue_indices, 'mbd_last']
        df_valid.loc[lastvalue_indices, 'y_pred'] = df_valid.loc[lastvalue_indices, f'select_rate{pred_m}_lag{pred_m}']
        
        # USE Trend.
        df_valid['mbd_trend'] = np.nan
        if self.USE_TREND:

            for category in ['high', 'low']:

                # pass if trend params don't exist.
                if not self.trend_params.get(f'{category}_trend_params').get(self.accum_cnt+1):
                    continue
                
                trend_params = self.trend_params.get(f'{category}_trend_params').get(self.accum_cnt+1).get('params')
                trend_method = self.trend_params.get(f'{category}_trend_params').get(self.accum_cnt+1).get('method')

                df_trend, trend_dict= preprocess.get_trend_dict(df_all, train_times, **trend_params)
                print(f'# of cfips that have {category} trend :', len(trend_dict))
                print('use method: ', trend_method)
                df_valid['mbd_trend'] = df_valid['y_base'] * df_valid['cfips'].map(trend_dict)

                if trend_method=='replace':
                    df_valid.loc[~df_valid['mbd_trend'].isna(), 'mbd_pred'] = df_valid.loc[~df_valid['mbd_trend'].isna(), 'mbd_trend']
                
                elif trend_method=='mean':
                    idx = (~df_valid['mbd_trend'].isna())&(~df_valid['mbd_model'].isna())
                    df_valid.loc[idx, 'mbd_pred'] = df_valid.loc[idx, 'mbd_trend'] * 0.75 + df_valid.loc[idx, 'mbd_model'] * 0.25
                    idx = (~df_valid['mbd_trend'].isna())&(df_valid['mbd_model'].isna())
                    df_valid.loc[idx, 'mbd_pred'] = df_valid.loc[idx, 'mbd_trend']
                
                else:
                    raise Exception('Wrong Trend Method.')
            
                df_valid.loc[~df_valid['mbd_trend'].isna(), 'y_pred'] = df_valid['cfips'].map(trend_dict) - 1
        
        df_valid['mbd_short_trend'] = np.nan
        if self.USE_SHORT_TREND:

            # pass if trend params don't exist.
            if self.trend_params.get(f'short_trend_params').get(self.accum_cnt+1):
          
                trend_params = self.trend_params.get(f'short_trend_params').get(self.accum_cnt+1).get('params')
                trend_method = self.trend_params.get(f'short_trend_params').get(self.accum_cnt+1).get('method')

                df_trend, trend_dict= preprocess.get_trend_dict(df_all, train_times, **trend_params)
                print('use method: ', trend_method)
                df_valid['mbd_short_trend'] = df_valid['y_base'] * df_valid['cfips'].map(trend_dict)

                idx = (~df_valid['mbd_short_trend'].isna())&(df_valid['mbd_trend'].isna())
                df_valid.loc[idx, 'mbd_pred'] = (df_valid.loc[idx, 'mbd_short_trend'] + df_valid.loc[idx, 'mbd_model']) / 2
                print(f'# of cfips that have only short trend :', sum(idx))

                if trend_method == 'mix':
                    idx = (~df_valid['mbd_short_trend'].isna())&(~df_valid['mbd_trend'].isna())
                    df_valid.loc[idx, 'mbd_trend'] = df_valid.loc[idx, 'mbd_trend'] * 0.5 + df_valid.loc[idx, 'mbd_short_trend'] * 0.5
                    df_valid.loc[idx, 'mbd_pred'] = df_valid.loc[idx, 'mbd_trend'] * 0.75 + df_valid.loc[idx, 'mbd_model'] * 0.25
                    print(f'# of cfips that have both short and long trend :', sum(idx))
            

        df_valid['mbd_season'] = np.nan
        df_valid['y_pred_season'] = np.nan
        if self.USE_SEASON:
            print('use SEASON.')

            df_season_dict = self.df_season[self.df_season['scale']==valid_time].set_index('cfips')['select_rate1_mean'].to_dict()
            df_valid['y_pred_season'] = df_valid['cfips'].map(df_season_dict)
            df_valid['mbd_season'] = (df_valid['y_pred_season'] + 1) * df_valid['y_base']
                        
            season_idx = (~df_valid['mbd_season'].isna())
            idx = season_idx&(df_valid['mbd_trend'].isna())
            df_valid.loc[idx, 'mbd_pred'] = (df_valid.loc[idx, 'mbd_model'] + df_valid.loc[idx, 'mbd_season']) / 2
            
            print(f'# of cfips season affected: ', sum(idx))
            
            if self.season_params['method']=='trend_mean':
            
                idx = season_idx&(~df_valid['mbd_trend'].isna())
                df_valid.loc[idx, 'mbd_pred'] = df_valid.loc[idx, 'mbd_model'] * 0.33 + df_valid.loc[idx, 'mbd_season'] * 0.33 + df_valid.loc[idx, 'mbd_trend'] * 0.34

                print(f'# of cfips season/trend/model mean: ', sum(idx))                

        df_valid['smape'] = utils.smape_arr(df_valid[mbd], df_valid['mbd_pred'])
        df_valid['smape_origin'] = utils.smape_arr(df_valid['mbd_origin'], df_valid['mbd_pred'])
        df_output = df_valid[self.output_features]
        
        return df_output


    def run_validation_for_pred_m(self, validation_times, pred_ms):
    
        df_output = pd.DataFrame(columns=self.output_features)
        for validation_time, pred_m in zip(validation_times, pred_ms):
            df = self.run_fit_predict(validation_time, pred_m)
            df_output = pd.concat([df, df_output])

        return df_output.reset_index().rename(columns={'index': 'row_id'}).set_index('row_id')


    def export_scores_summary(self, pred_ms = [1,2,3,4,5], scale=[], filename=''):
        
        if not self.output_dic:
            raise Exception('found no result. Execute run_validation first.')
        
        row = len(self.output_dic)
        output_array = np.zeros((6, 2))
        for pred_m in pred_ms:
            df = self.output_dic[pred_m]
    
            if scale:
                df = df[df['scale'].isin(scale)]

            output_array[pred_m-1] = df.groupby('scale')['smape_origin'].mean().describe()[['mean', 'std']].to_numpy()

        dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        dt_str = dt_now.strftime('%Y-%m-%d_%H:%M:%S')
        name = self.run_fold_name
        if filename:
            name += f'_{filename}'
        name = self.run_fold_name + f'_{dt_str}'

        df = pd.DataFrame(output_array, columns=['mean', 'std'], index=list(range(1, 7)))
        df.loc[6, 'mean'] = df.loc[2:4, 'mean'].mean()
        df.loc[6, 'std'] = df.loc[3:5, 'mean'].mean()

        if self.save_path:
            df.to_csv(f'../output/{name}.csv')
        else:
            df.to_csv(f'{name}.csv')
        
        if self.save_output_dic:
            if self.save_path:
                utils.save_pickle(self.output_dic, f'{name}')
            else:
                save_path = f'{name}.pickle'
                with open(save_path, 'wb') as f:
                    pickle.dump(self.output_dic, f)
            print(f'saved {name}.pickle')

        print(f'saved output/{name}.csv')


    def run_validation(self, max_month=40, m_len=5, pred_ms=[1,2,3,4,5], export=True, 
                        filename='', accum_cnt=0, out_idx=[]):

        self.accum_cnt = accum_cnt
        if accum_cnt:
            self.get_df_all_dict(list(range(max_month-m_len, max_month)), smooth=False)
            print('success in updating df_all_dict')
        
        if not out_idx:
            out_idx = pred_ms
        
        validation_times = [max_month - i for i in range(m_len)]
        output_dic = dict()
        for pred_m, out_id in zip(pred_ms, out_idx):
            pred_m_len = [pred_m] * m_len
            df_output = self.run_validation_for_pred_m(validation_times, pred_m_len)
            output_dic[out_id] = df_output
            print(f'saved output_dic[{out_id}].')
        
        self.output_dic.update(output_dic)

        if export:
            self.export_scores_summary(pred_ms=pred_ms, filename=filename)


    def accum_validation(self, max_month=40, max_pred_m = 5, m_len=5, export=True):

        if self.USE_SEASON:
            max_train = max_month - m_len - max_pred_m + 1
            abs_thre = self.season_params['abs_thre']
            active_thre = self.season_params['active_thre']
            v_clip = self.season_params['v_clip']
            self.df_season = utils.create_df_season(self.df_all, validate=True,
                abs_thre=abs_thre, active_thre=active_thre, v_clip = v_clip)
        
        self.run_validation(
            max_month=max_month,
            m_len=m_len+3,
            pred_ms=[1],
            export=False
        )

        for i in range(2, max_pred_m+1):
            self.run_validation(
                max_month=max_month,
                m_len=m_len + (4-i),
                pred_ms=[1],
                accum_cnt=i-1,
                out_idx=[i],
                export=False
            )

        if export:
            self.export_scores_summary(pred_ms=[i for i in range(1, max_pred_m+1)], scale=[36,37,38,39,40])


    def create_submission(self, target_scale=[41,42,43,44,45], save=True, filename=''):

        if self.USE_SEASON:
            abs_thre = self.season_params['abs_thre']
            active_thre = self.season_params['active_thre']
            v_clip = self.season_params['v_clip']
            
            self.df_season = utils.create_df_season(self.df_all, validate=False, 
                active_thre=active_thre, abs_thre=abs_thre, v_clip=v_clip)

        self.run_validation(
            max_month=target_scale[0],
            m_len=1,
            pred_ms=[1],
            export=False
        )
        df_pred = self.output_dic[1]

        for k, i in enumerate(target_scale[1:]):
            self.run_validation(
                max_month=i,
                m_len=1,
                pred_ms=[1],
                accum_cnt=k+1,
                out_idx=[k+2],
                export = False
            )
            df_pred = pd.concat([df_pred, self.output_dic[k+2]])
        
        # 予測値をマージ
        df_merged = pd.merge(self.df_subm, df_pred['mbd_pred'], how='left', on='row_id')
        df_merged.loc[~df_merged['mbd_pred'].isna(), mbd] = df_merged['mbd_pred']
        df_submission = df_merged[mbd]

        # 人口分だけ補正
        if self.start_max_scale==41: # 1月分は補正不要
            df_submission = utils.adjust_population(df_submission, self.df_census, start_month='2023-02-01')
        else:
            df_submission = utils.adjust_population(df_submission, self.df_census)

        # round_to_integer
        df_submission = utils.round_integer(df_submission, self.df_census)
        
        dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        dt_str = dt_now.strftime('%Y-%m-%d_%H:%M:%S')
        if save:
            name = self.run_fold_name
            if filename:
                name += f'_{filename}'
            name += f'_{dt_str}'

            if self.save_path:
                df_submission.to_csv(f'../submission/{name}.csv')
            else:
                df_submission.to_csv(f'{name}.csv')

            print(f'saved {name}')
            
        self.df_pred = df_pred
        self.df_merged = df_merged
        self.df_submission = df_submission