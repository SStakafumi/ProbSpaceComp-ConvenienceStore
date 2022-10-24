# 提出ファイルについて

* sub_by_lgb_by_kotrying.csv
    * socre 0.8くらい。が、Publicスコアが1.3とかなのでvalidationの枠組みを間違えてリークしている可能性がある。
* sub_by_lgb_by_matsuyama_gridsearch_only.csv
    * max_depth, n_estimator, min_child_samples, colsample_bytreeでグリッドサーチした
    * socre 2.006で微妙
* sub_by_lgb_by_kotrying_max_depth_5.csv
    * グリッドサーチがうまくいかなかったので、上のプログラムをmax_depth=5でやり直し
    * score 1.694
* sub_lgb_grid_max_min_cs_sincos_rbf_by_matsuyama.csv
    * [Nvidia Technical Blog](https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/) を参考にsincos変換とrbf変換つかった。
    * sin,cos
        * day_of_year
        * month
        * weekday
    * rbf
        * day_of_yearを12分割
    * sin, cos, rbf以外の情報(id, date, time weekday等)は訓練データに使わなかった。
    * 検証データは最後２ヶ月(２月、３月)
    * score 0.9くらい
* sub_lgb_max_dp_4_sincos_rbf_by_matsuyama.csv
    * grid search やめて max_depth 4
    * score 0.99
* sub_lgb_kotrying_sincos_rbf_by_matsuyama_valid_23.csv
    * kotrying にsincos, rbfを追加
    * 2, 3月を valid
    * score 0.95 -> 1.566
* sub_lgb_kotrying_sincos_rbf_RFE06_valid_34_by_matsuyama.csv
    * 4, 3月をvalid
    * score 0.80 -> 1.506
* 
ファイル名	提出者	ステータス	提出時間	Publicスコア	Privateスコア	メッセージ	操作	最終提出
* sub_lgb_kotrying_plus_day_of_year_valid3_max_depth5.csv
    * 基本 kotrying参照
    * max_depth = 5
    * valid は3月
    * day_of_year 追加
    * score 0.7412706596597354 -> 1.206
    * RFE なし
* sub_lgb_lag_first_valid3_remove_timeinfo.csv
    * lgbm 時間情報なしでラグ特徴量を作成(気温、降水量の情報はあり
    * 下でアンサンブル
    * ラグ特徴量のスコアは 0.973
* sub_lgb_lag_first_valid3_remove_timeinfo_and_lgbbase_ansamble.csv
    * lgbm baseline + lgbm 時間情報なしでラグ特徴量を作成(気温、降水量の情報はあり)のアンサンブル
    * score 1.212
    * valid 3月
* sub_lgb_lag_first_valid3.csv
    * ラグ特徴量＋baselineで使ったデータ全部(日付ごとに20個モデル作った)
        * 過去10週の7日前
        * 上記の過去5週分の平均
        * 過去20日
    * lgbm
    * score 0.965 -> 1.221