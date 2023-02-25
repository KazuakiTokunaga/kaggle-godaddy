import datetime
import pandas as pd
import numpy as np

from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from modules import utils
from modules import preprocess
from sklearn.pipeline import Pipeline

mbd = 'microbusiness_density'


def get_model(algo='lgbm'):

    if algo == 'lgbm':
        print('use lgbm model.')
        params = {
            'n_iter': 200,
            'verbosity': -1,
            'objective': 'l1',
            'random_state': 42,
            'colsample_bytree': 0.8841279649367693,
            'colsample_bynode': 0.10142964450634374,
            'max_depth': 8,
            'learning_rate': 0.013647749926797374,
            'lambda_l1': 1.8386216853616875,
            'lambda_l2': 7.557660410418351,
            'num_leaves': 61,
            "seed": 42,
            'min_data_in_leaf': 213
        }
        model = lgb.LGBMRegressor(**params)

        return model
    

    elif algo=='xgb':
        print('use xgb model.')
        model = xgb.XGBRegressor(
            objective='reg:pseudohubererror',
            #objective='reg:squarederror',
            tree_method="hist",
            n_estimators=805,
            learning_rate=0.0075,
            max_leaves = 31,
            subsample=0.60,
            colsample_bytree=0.50,
            max_bin=4096,
            n_jobs=2,
            eval_metric='mae',
        )

        return model


    elif algo=='ensemble':
        print('use ensemble model.')
        params = {
        'n_iter': 200,
        'verbosity': -1,
        'objective': 'l1',
        'random_state': 42,
        'colsample_bytree': 0.8841279649367693,
        'colsample_bynode': 0.10142964450634374,
        'max_depth': 8,
        'learning_rate': 0.013647749926797374,
        'lambda_l1': 1.8386216853616875,
        'lambda_l2': 7.557660410418351,
        'num_leaves': 61,
        "seed": 42,
        'min_data_in_leaf': 213}

        lgb_model = lgb.LGBMRegressor(**params)
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:pseudohubererror',
            #objective='reg:squarederror',
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
            iterations=1200,
            loss_function="MAPE",
            verbose=0,
            learning_rate=0.075,
            l2_leaf_reg=0.2,
            subsample=0.50,
            max_bin=4096,
        )
        
        return VotingRegressor([
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cat', cat_model)
        ])


class LgbmBaseline():

    def __init__(self, run_fold_name, df_subm, df_all, df_census, start_all_dict, save_path=True, params={
        "act_thre": 2.00,
        "abs_thre": 1.00,
        "USE_LAG": 5,
        "USE_TREND": False,
        "blacklist": [],
        "blacklistcfips": [],
        "clip": (None, None),
        "model": 'lgbm'
    }):

        self.run_fold_name = run_fold_name
        self.df_subm = df_subm
        self.df_census = df_census
        self.output_dic = '../output/'

        self.act_thre = params['act_thre']
        self.abs_thre = params['abs_thre']
        self.USE_LAG = params['USE_LAG']
        self.USE_TREND = params['USE_TREND']
        self.blacklist = params['blacklist']
        self.blacklistcfips = params['blacklistcfips']
        self.clip = params['clip']
        self.model = params['model']
        
        self.save_path = save_path
        self.print_feature = False
        self.accum = False
        self.output_dic = dict()

        self.df_all_dict = dict()
        for i in range(start_all_dict, 41):
            self.df_all_dict[i] = preprocess.add_lag_features(df_all, max_scale=i, USE_LAG = self.USE_LAG)
            print(f'created df_all_dict[{i}]')

        self.df_all = self.df_all_dict[40]
        self.output_features = ['cfips', 'county', 'state', 'state_i', 'microbusiness_density', 'active', 'year','month', 'scale', 
                                'mbd_pred', 'mbd_model', 'mbd_last', 'mbd_trend', 'y_base', 'y_pred', 'smape']


    def get_df_all_dict(self, train_times, accum_cnt):
        
        for i in train_times:
            print(f'create df_all_dict[{i}].')
            output1 = self.output_dic[accum_cnt]
            r1 = output1[output1['scale']==i].reset_index()
            for c in range(1, accum_cnt):
                output2 = self.output_dic[accum_cnt-c]
                r2 = output2[output2['scale']==i-c].reset_index()
                r1 = pd.concat([r1, r2])
            
            last_exist_scale = i - accum_cnt

            df_all_t = self.df_all
            df_merged = df_all_t.merge(r1[['row_id', 'mbd_pred']], how='left', on='row_id').set_index('row_id')
            df_merged.loc[~df_merged['mbd_pred'].isna(), mbd] = df_merged.loc[~df_merged['mbd_pred'].isna(), 'mbd_pred']
            df_merged.drop(['mbd_pred'], axis=1, inplace=True)
            
            idx = df_merged['scale']>last_exist_scale
            df_merged.loc[idx, 'active'] =  (df_merged.loc[idx, f'select_lastactive{last_exist_scale}'] / df_merged.loc[idx, f'select_lastmbd{last_exist_scale}']) * df_merged.loc[idx, mbd]
            df_merged.loc[df_merged['active'].isna(), 'active'] = 0

            df_all_t = preprocess.add_lag_features(df_merged, max_scale=i, USE_LAG=self.USE_LAG)
            
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
        model = get_model(algo=self.model)
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
        # lastvalue_indices = (~df_valid['cfips'].isna())
        lastvalue_indices = ~(valid_indices)
        df_valid.loc[lastvalue_indices, 'mbd_pred'] = df_valid.loc[lastvalue_indices, 'mbd_last']
        df_valid.loc[lastvalue_indices, 'y_pred'] = df_valid.loc[lastvalue_indices, f'select_rate{pred_m}_lag{pred_m}']
        
        # USE Trend.
        df_valid['mbd_trend'] = np.nan
        if self.USE_TREND and pred_m == 1:
            df_trend, trend_dict= preprocess.get_trend_dict(df_all, train_times)
            print('# of cfips that have trend :', len(trend_dict))
            for cfip in trend_dict:
                df_valid.loc[df_valid['cfips']==cfip, 'mbd_trend'] = df_valid.loc[df_valid['cfips']==cfip, 'y_base'] * trend_dict[cfip]
                df_valid.loc[df_valid['cfips']==cfip, 'mbd_pred'] = df_valid.loc[df_valid['cfips']==cfip, 'mbd_trend']
                df_valid.loc[df_valid['cfips']==cfip, 'y_pred'] = (trend_dict[cfip] - 1)
        
        df_valid['smape'] = utils.smape_arr(df_valid['microbusiness_density'], df_valid['mbd_pred'])
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

            output_array[pred_m-1] = df.groupby('scale')['smape'].mean().describe()[['mean', 'std']].to_numpy()

        dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        dt_str = dt_now.strftime('%Y-%m-%d_%H:%M:%S')
        name = self.run_fold_name
        if filename:
            name += f'_{filename}'
        name = self.run_fold_name + f'_{dt_str}'

        df = pd.DataFrame(output_array, columns=['mean', 'std'], index=list(range(1, 7)))

        if self.save_path:
            df.to_csv(f'../output/{name}.csv')
        else:
            df.to_csv(f'{name}.csv')

        print(f'saved output/{name}.csv')


    def run_validation(self, max_month=40, m_len=5, pred_ms=[1,2,3,4,5], export=True, 
                        filename='', accum = False, accum_cnt=0, out_idx=[]):

        self.accum = accum
        if accum:
            self.get_df_all_dict(list(range(max_month-m_len, max_month)), accum_cnt)
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
                accum=True,
                accum_cnt=i-1,
                out_idx=[i],
                export=False
            )

        if export:
            self.export_scores_summary(pred_ms=[i for i in range(1, max_pred_m+1)], scale=[36,37,38,39,40])


    def create_submission(self, target_scale=[41,42,43,44,45], accum=True, save=True, filename=''):

        if accum:
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
                    accum=True,
                    accum_cnt=k+1,
                    out_idx=[k+2],
                    export = False
                )
                df_pred = pd.concat([df_pred, self.output_dic[k+2]])
        
        else:
            for k, i in enumerate(target_scale):
                self.run_validation(max_month=i, m_len=1, pred_ms=[k+1], export=False)
                if not k:
                    df_pred = self.output_dic[k+1]
                else:
                    df_pred = pd.concat([df_pred, self.output_dic[k+1]])
        
        # 予測値をマージ
        df_merged = pd.merge(self.df_subm, df_pred['mbd_pred'], how='left', on='row_id')
        df_merged.loc[~df_merged['mbd_pred'].isna(), mbd] = df_merged['mbd_pred']
        df_submission = df_merged[mbd]

        # 人口分だけ補正
        df_submission = df_submission.reset_index()
        df_submission['cfips'] = df_submission['row_id'].apply(lambda x: int(x.split('_')[0]))
        adult2020 = self.df_census.set_index('cfips')['adult_2020'].to_dict()
        adult2021 = self.df_census.set_index('cfips')['adult_2021'].to_dict()
        df_submission['adult2020'] = df_submission['cfips'].map(adult2020)
        df_submission['adult2021'] = df_submission['cfips'].map(adult2021)
        df_submission[mbd] = df_submission[mbd] * df_submission['adult2020'] / df_submission['adult2021']
        df_submission = df_submission.drop(['adult2020','adult2021','cfips'],axis=1)
        df_submission = df_submission.set_index('row_id')
        
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