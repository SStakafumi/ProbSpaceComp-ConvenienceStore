# Library
import os
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import statsmodels.api as sm
import lightgbm as lgb
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss
from lightgbm import LGBMRegressor
from tqdm.auto import tqdm

import warnings
warnings.simplefilter('ignore')


seed = 42

train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')
submission_df = pd.read_csv('data/submission.csv')


def preprocessing(df, mode='train'):
    df_tmp = df.copy()
    input_year = 2021

    df_tmp['time'] = pd.to_datetime(df_tmp.date, format='%m/%d')
    df_tmp['year'] = df_tmp['time'].dt.year
    df_tmp['month'] = df_tmp['time'].dt.month
    df_tmp['day'] = df_tmp['time'].dt.day
    if mode == 'train':
        df_tmp.loc[df_tmp['month'] > 3, 'year'] = input_year
        df_tmp.loc[df_tmp['month'] <= 3, 'year'] = input_year + 1
    else:
        df_tmp['year'] = input_year + 1
    df_tmp['time'] = pd.to_datetime(
        {'year': df_tmp.year, 'month': df_tmp.month, 'day': df_tmp.day})
    df_tmp['weekday'] = df_tmp['time'].dt.weekday
    return df_tmp


train_df = preprocessing(train_df, mode='train')
test_df = preprocessing(test_df, mode='test')


target_columns = ['ice1', 'ice2', 'ice3',
                  'oden1', 'oden2', 'oden3', 'oden4', 'hot1', 'hot2', 'hot3', 'dessert1',
                  'dessert2', 'dessert3', 'dessert4', 'dessert5', 'drink1', 'drink2',
                  'drink3', 'drink4', 'drink5', 'drink6', 'alcol1', 'alcol2', 'alcol3',
                  'snack1', 'snack2', 'snack3', 'bento1', 'bento2', 'bento3', 'bento4',
                  'tild1', 'tild2', 'men1', 'men2', 'men3', 'men4', 'men5', 'men6']

# 検証データのindexを指定(訓練データの最後2ヶ月を使用)
valid_index = range(297, 351)  # month:2,3
# valid_index = range(325,351) # month:3

# 予測結果を保存する辞書型データ
results = dict({})
all_lgb_score = []

# 商品ごとの予測を作成

for c in tqdm(target_columns):
    train_tmp = train_df.copy()
    test_tmp = test_df.copy()

    # ice
    if c in ['ice1', 'ice2', 'ice3']:
        # 予測期間はアイスが人気な7, 8 ,9月ではないので除外
        train_tmp = train_tmp[~train_tmp['month'].isin([7, 8, 9])]
        # アイスは金曜に人気
        train_tmp['is_wday4'] = train_df['weekday'].isin([4]).astype(int)
        test_tmp['is_wday4'] = test_df['weekday'].isin([4]).astype(int)
    # elif c in ['ice2']:
    #         pass
    # oden
    elif c in ['oden1', 'oden2', 'oden3', 'oden4']:
        # おでんやって無い夏は考えない
        train_tmp = train_tmp[train_tmp['month'].isin(
            [10, 11, 12, 1, 2, 3, 4, 5])]
        # おでんは水木で人気
        train_tmp['is_wday23'] = train_df['weekday'].isin([2, 3]).astype(int)
        test_tmp['is_wday23'] = test_df['weekday'].isin([2, 3]).astype(int)
    # hot
    elif c in ['hot1', 'hot2', 'hot3']:
        # ホットスナックは月、火、金で不人気
        train_tmp['is_wday014'] = train_df['weekday'].isin(
            [0, 1, 4]).astype(int)
        test_tmp['is_wday014'] = test_df['weekday'].isin([0, 1, 4]).astype(int)
        # pass
    # dessert
    elif c in ['dessert1', 'dessert2', 'dessert3', 'dessert4', 'dessert5']:
        # デザートは水曜と日曜で人気
        train_tmp['is_wday36'] = train_df['weekday'].isin([3, 6]).astype(int)
        test_tmp['is_wday36'] = test_df['weekday'].isin([3, 6]).astype(int)
        # pass
    # drink1456
    elif c in ['drink1', 'drink4', 'drink5', 'drink6']:
        # 夏に人気なドリンクは夏のデータは使わない
        train_tmp = train_tmp[~train_tmp['month'].isin([7, 8, 9])]
        # 金曜に人気
        train_tmp['is_wday4'] = train_df['weekday'].isin([4]).astype(int)
        test_tmp['is_wday4'] = test_df['weekday'].isin([4]).astype(int)
    # drink23
    elif c in ['drink2', 'drink3']:
        # 常に人気が変わらないドリンクは全期間つかう。火曜に人気
        train_tmp['is_wday1'] = train_df['weekday'].isin([1]).astype(int)
        test_tmp['is_wday1'] = test_df['weekday'].isin([1]).astype(int)
    # alcohol
    elif c in ['alcol1', 'alcol2', 'alcol3']:
        # 酒は水木で人気
        train_tmp['is_wday23'] = train_df['weekday'].isin([2, 3]).astype(int)
        test_tmp['is_wday23'] = test_df['weekday'].isin([2, 3]).astype(int)
    # snack
    elif c in ['snack1', 'snack2', 'snack3']:
        train_tmp['is_wday0'] = train_df['weekday'].isin([0]).astype(int)
        train_tmp['is_wday14'] = train_df['weekday'].isin([1, 4]).astype(int)
        test_tmp['is_wday0'] = test_df['weekday'].isin([0]).astype(int)
        test_tmp['is_wday14'] = test_df['weekday'].isin([1, 4]).astype(int)
    # bento
    elif c in ['bento1', 'bento2', 'bento3', 'bento4']:
        # 弁当は月、火、金で不人気
        train_tmp['is_wday014'] = train_df['weekday'].isin(
            [0, 1, 4]).astype(int)
        test_tmp['is_wday014'] = test_df['weekday'].isin([0, 1, 4]).astype(int)
    # tild
    elif c in ['tild1', 'tild2']:
        # 日曜に人気
        train_tmp['is_wday6'] = train_df['weekday'].isin([6]).astype(int)
        test_tmp['is_wday6'] = test_df['weekday'].isin([6]).astype(int)
    # men
    elif c in ['men1', 'men2', 'men3', 'men4', 'men5', 'men6']:
        train_tmp['is_wday014'] = train_df['weekday'].isin(
            [0, 1, 4]).astype(int)
        test_tmp['is_wday014'] = test_df['weekday'].isin([0, 1, 4]).astype(int)

    # id, rain, highest, lowest, year, month, day, weekday, 自分で付け加えた特徴量
    train_columns = [
        c for c in train_tmp.columns if c not in target_columns if c not in ['date', 'time']]

    X_train = train_tmp[~train_tmp['id'].isin(valid_index)][train_columns]
    y_train = train_tmp[~train_tmp['id'].isin(valid_index)][c]
    X_valid = train_tmp[train_tmp['id'].isin(valid_index)][train_columns]
    y_valid = train_tmp[train_tmp['id'].isin(valid_index)][c]
    X_test = test_tmp[train_columns]

    # グリッドサーチ用パラメータ
    param_space = {"n_estimators": [100, 500, 1000, 10000],  # 2000まではいらない
                   "max_depth": [3, 7, 10, 15],  # ,20,30は過学習を引き起こす
                   "colsample_bytree": [0.5, 0.7, 0.9],
                   "min_child_samples": [10, 15, 20, 25, 30]}

    # 分位点
    qs = np.array([0.01, 0.1, 0.5, 0.9, 0.99])

    # 分位点ごとのスコア
    q_scores = []

    # 分位点ごとにモデルを作成
    for q in qs:
        # グリッドサーチ

        # グリッドサーチによるスコア、パラメータ、モデル
        scores = []
        params = []
        models = []

        param_combinations = itertools.product(param_space['n_estimators'],
                                               param_space['max_depth'],
                                               param_space['colsample_bytree'],
                                               param_space['min_child_samples'])

        for n_est, max_dp, col_bt, min_cs in param_combinations:
            print(f'{c}_{q}_{n_est}_{max_dp}_{col_bt}_{min_cs}')
            lgb = LGBMRegressor(
                objective='quantile',
                alpha=q,
                random_state=seed,
                n_estimators=n_est,
                max_depth=max_dp,
                colsample_bytree=col_bt,
                min_child_samples=min_cs
            )

            # 学習
            lgb.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                    early_stopping_rounds=100, verbose=False)
            score = lgb.best_score_['valid_0']['quantile']

            # グリッドサーチの結果
            params.append((n_est, max_dp, col_bt, min_cs))
            models.append(lgb)
            scores.append(score)

        # 最もスコアが良いものをベストなパラメータにする
        best_idx = np.argsort(scores)[0]
        best_score = scores[best_idx]
        best_param = params[best_idx]
        best_model = models[best_idx]

        # 最も良いモデルで予測
        y_pred = best_model.predict(X_test)
        results[(c, q)] = y_pred

        q_scores.append(best_score)

    all_lgb_score.append(q_scores)

score_df = pd.DataFrame(np.array(all_lgb_score),
                        columns=qs, index=target_columns)

print(f'ave score: {np.array(all_lgb_score).mean()}')
