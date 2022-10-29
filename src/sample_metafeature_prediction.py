import os
import os.path as osp
import sys
from datetime import date
import random
from pprint import pprint
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.neighbors import NearestNeighbors
from sklego.preprocessing import RepeatingBasisFunction
import statsmodels.api as sm
from statsmodels.tsa import stattools, ar_model
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
import optuna
import optuna.integration.lightgbm as lgb_opt
from tqdm import tqdm
import joblib


def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Data setup
DATA_DIR = "./data"
tmp_year = 2018

train_df = pd.read_csv(osp.join(DATA_DIR, "train_data.csv"))
test_df = pd.read_csv(osp.join(DATA_DIR, "test_data.csv"))
train_df["date"] = str(tmp_year) + "/" + train_df["date"]
test_df["date"] = str(tmp_year+1) + "/" + test_df["date"]

train_start_date = "2018/4/11"
train_end_date = "2019/3/26"
test_start_date = "2019/3/27"
test_end_date = "2019/4/16"

train_df["date"] = pd.date_range(start=train_start_date, end=train_end_date)
test_df["date"] = pd.date_range(start=test_start_date, end=test_end_date)
train_df.set_index("date", inplace=True)
test_df.set_index("date", inplace=True)

SUBMISSION_COLS = pd.read_csv(osp.join(DATA_DIR, "submission.csv")).columns
IRREGULAR_COLS = [f"oden{i}" for i in range(1, 5)]
EXPLANETORY_COLS = sorted(["highest", "lowest", "rain", "id"])
ADF_TARGET_COLS = sorted(list(set(train_df.columns) - set(EXPLANETORY_COLS) - set(IRREGULAR_COLS)))
UPDATED_ALMOST_STATIONARY_COLS = [
    "dessert1",
    "dessert2",
    "dessert3",
    "dessert4",
    "dessert5",
    "alcol1",
    "alcol2",
    "alcol3",
    "snack2",
    "bento1",
    "bento2",
    "bento3",
    "bento4",
    "men1",
    "men2",
    "men3",
    "men5",
]
QUANTILES = [0.01, 0.1, 0.5, 0.9, 0.99]
TEMPLATE_EXPLANETORY_COLS = ["highest", "lowest", "rain", "day_of_year", "day_of_week", "month"]


# =============================================================================
# Feature engineering start
# =============================================================================
train_df["month"] = train_df.index.month
test_df["month"] = test_df.index.month
train_df["day_of_week"] = train_df.index.day_of_week
test_df["day_of_week"] = test_df.index.day_of_week
train_df["day_of_year"] = train_df.index.day_of_year
test_df["day_of_year"] = test_df.index.day_of_year

# Cyclical Encoding
train_df["month sin"] = sin_transformer(12).fit_transform(train_df.index.month)
train_df["month cos"] = cos_transformer(12).fit_transform(train_df.index.month)
train_df["day sin"] = sin_transformer(7).fit_transform(train_df.index.day_of_week)
train_df["day cos"] = cos_transformer(7).fit_transform(train_df.index.day_of_week)

test_df["month sin"] = sin_transformer(12).fit_transform(test_df.index.month)
test_df["month cos"] = cos_transformer(12).fit_transform(test_df.index.month)
test_df["day sin"] = sin_transformer(7).fit_transform(test_df.index.day_of_week)
test_df["day cos"] = cos_transformer(7).fit_transform(test_df.index.day_of_week)

# RBF Encoding
rbf_month = RepeatingBasisFunction(n_periods=12,
                         	column="day_of_year",
                         	input_range=(1,365),
                         	remainder="drop")
rbf_month.fit(train_df)
train_df_radial = pd.DataFrame(index=train_df.index,
               	data=rbf_month.transform(train_df))

train_df[[f"day of year radial {i}" for i in range(12)]] = train_df_radial

rbf_month.fit(test_df)
test_df_radial = pd.DataFrame(index=test_df.index,
               	data=rbf_month.transform(test_df))

test_df[[f"day of year radial {i}" for i in range(12)]] = test_df_radial


# =============================================================================
# Start trainig ARMA model
# =============================================================================
each_stationary_sarima_models = {}
for target_col in tqdm(UPDATED_ALMOST_STATIONARY_COLS):
    print(target_col)
    endog = train_df.loc[:, target_col]
    # exog = sm.add_constant(train_df.loc[:"2019", "rain"])
    nobs = endog.shape[0]

    select = ar_model.ar_select_order(endog, maxlag=100, trend="ct")
    
    if len(select.ar_lags) == 0:
        selected_lags = [1, 2, 7, 14, 21, 28, 35]
        print(target_col, "col selected ar lag is empty")
    else:
        selected_lags = select.ar_lags
        
    
    # model = sm.tsa.statespace.SARIMAX(endog, order=(30, 0, 5), seasonal_order=(1, 1, 0, 7))
    model = sm.tsa.statespace.SARIMAX(
        endog, 
        order=(selected_lags, 0, 5),
        enforce_stationarity=False,
        enforce_invertibility=False,
        )
    # model = sm.tsa.statespace.SARIMAX(endog, exog=exog, ordor=(15, 0, 0), seasonal_order=(1, 1, 0, 7))
    fit_res = model.fit(disp=True)
    # print(fit_res.summary())
    
    
    predict = fit_res.get_prediction()
    predict_ci = predict.conf_int()
    
    # Dynamic predictions
    predict_by = fit_res.get_prediction(dynamic='2019-03-01')
    predict_by_ci = predict_by.conf_int()
    
    # # Graph
    # fig, ax = plt.subplots(figsize=(20,4))
    # # Plot data points
    # train_df.loc['2019-03-01':, target_col].plot(ax=ax, style='o', label='Observed')

    # # Plot predictions
    # predict.predicted_mean.loc['2019-03-01':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
    # ci = predict_ci.loc['2019-03-01':]
    # ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
    # predict_by.predicted_mean.loc['2019-03-01':].plot(ax=ax, style='g', label='Dynamic forecast')
    # ci = predict_by_ci.loc['2019-03-01':]
    # ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

    # legend = ax.legend(loc='lower right')
    # plt.show()
    
    # Save each model
    each_stationary_sarima_models[target_col] = fit_res

# Add ARMA prediction cols
test_df_meta_feature = test_df.copy()
train_df_meta_feature = train_df.copy()
for col, each_model in each_stationary_sarima_models.items():
    train_df_meta_feature[col + "_arima"] = each_model.predict()
    test_pred = each_model.forecast(len(test_df_meta_feature))
    test_df_meta_feature[col + "_arima"] = test_pred.values


# =============================================================================
# Start trainig LGBM model with tuning
# =============================================================================
# NOTE: LGBM with parameter tuning takes long time. You can substitute normal LGBM by replacing lgb_opt to lgb
trained_lgbm_models = {}
for each_col, each_quantile in tqdm(product(ADF_TARGET_COLS, QUANTILES)):
    # lgbm_model = lgb.LGBMRegressor(objective="quantile", alpha=each_quantile, boosting_type="dart")
    if (each_col+"_arima") in train_df_meta_feature.columns:
        explanetory_cols = TEMPLATE_EXPLANETORY_COLS.copy() + [each_col+"_arima"]
        # print("Use arima")
    else:
        explanetory_cols = TEMPLATE_EXPLANETORY_COLS.copy()
    
    train_X = train_df_meta_feature.loc[:"2019", explanetory_cols]
    train_y = train_df_meta_feature.loc[:"2019", each_col]
    valid_X = train_df_meta_feature.loc["2019":, explanetory_cols]
    valid_y = train_df_meta_feature.loc["2019":, each_col]
    test_X = test_df_meta_feature.loc[:, explanetory_cols]
    
    dtrain = lgb.Dataset(train_X, label=train_y)
    dvalid = lgb.Dataset(valid_X, label=valid_y)
    
    params = {
        "objective": "quantile",
        "metric": "quantile",
        "alpha": each_quantile,
        "boosting": "dart",
        # "boosting": "gbdt",
        "random_seed": 42,
        "verbose": -1,
    }
    
    verbose_eval = -1
    best_params, tuning_history = dict(), list()
    lgbm_model = lgb_opt.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=10000,
        callbacks=[
            # lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(verbose_eval)
            ],
        # best_params=best_params,
        # tuning_history=tuning_history,
    )
    # lgbm_model.fit(train_X, train_y)
    
    trained_lgbm_models[each_col+"_"+str(each_quantile)] = lgbm_model
    
    pred_y = lgbm_model.predict(valid_X)
    pinball_loss = mean_pinball_loss(valid_y, pred_y, alpha=each_quantile)
    # print(each_col, each_quantile, pinball_loss)
    
    test_pred_y = lgbm_model.predict(test_X)
    test_df_meta_feature[each_col + f"_{each_quantile}"] = test_pred_y
    

oden_nonzero_df = train_df[train_df[IRREGULAR_COLS].sum(axis=1) > 0]
for oden_col, each_quantile in product(IRREGULAR_COLS, QUANTILES):
    train_end = "2019-02"
    train_X = oden_nonzero_df.loc[:train_end, TEMPLATE_EXPLANETORY_COLS]
    train_y = oden_nonzero_df.loc[:train_end, oden_col]
    valid_X = oden_nonzero_df.loc[train_end:, TEMPLATE_EXPLANETORY_COLS]
    valid_y = oden_nonzero_df.loc[train_end:, oden_col]
    test_X = test_df_meta_feature.loc[:, TEMPLATE_EXPLANETORY_COLS]
    
    dtrain = lgb.Dataset(train_X, label=train_y)
    dvalid = lgb.Dataset(valid_X, label=valid_y)
    
    params = {
        "objective": "quantile",
        "metric": "quantile",
        "alpha": each_quantile,
        # "boosting": "dart",
        "boosting": "gbdt",
        "random_seed": 42,
        "verbose": -1,
    }
    
    verbose_eval = -1
    lgbm_model = lgb_opt.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=10000,
        callbacks=[
            # lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(verbose_eval)
            ]
    )
    # lgbm_model.fit(train_X, train_y)
    
    trained_lgbm_models[oden_col+"_"+str(each_quantile)] = lgbm_model
    
    pred_y = lgbm_model.predict(valid_X)
    pinball_loss = mean_pinball_loss(valid_y, pred_y, alpha=each_quantile)
    print(oden_col, each_quantile, pinball_loss)
    
    test_pred_y = lgbm_model.predict(test_X)
    test_df_meta_feature[oden_col + f"_{each_quantile}"] = test_pred_y


# =============================================================================
# Save outputs and trained models.
# =============================================================================
dt_now = datetime.now()
test_df_meta_feature[SUBMISSION_COLS].to_csv(dt_now.strftime("%m_%d_%H:%M.csv"), index=False)
joblib.dump(trained_lgbm_models, "trained_models_dart.bin")