# sklearn.feature_selection.RFE

再帰的特徴量削減。以下のステップで特徴量を削減

1. 全特徴量で訓練しモデルを生成
2. 最低のFeature Importnace/ Coef の特徴量を1つ減らして訓練しモデルを生成
3. 2のステップを残す特徴量の数に達するまで繰り返し


参照1: [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
参照2: [特徴量選択の種類](https://qiita.com/FukuharaYohei/items/db88a8f4c4310afb5a0d#21-sfssequential-feature-selection-%E9%80%90%E6%AC%A1%E7%89%B9%E5%BE%B4%E9%81%B8%E6%8A%9E)

特徴量（例えば線形モデルの係数）に重みを与える外部推定器があるとき，再帰的特徴除去（RFE）の目的は，より小さな特徴量の集合を再帰的に考慮することによって特徴量を選択することである．まず、推定器は特徴の初期セットで学習され、各特徴の重要度は、任意の特定の属性またはcallableによって取得される。次に、現在の特徴量から最も重要度の低い特徴量を切り出す。この手順は、最終的に選択すべき特徴量の数に達するまで、刈り込まれた集合に対して再帰的に繰り返される。

## パラメータ
* estimator: Estimator instance
    * 特徴量の重要度に関する情報(coef_, feature_importances_など)を提供するフィット法を持つ教師あり学習推定器です。
* n_features_to_select: int or float, default=None
    * 選択するフィーチャーの数。Noneの場合、半分のフィーチャーが選択される。整数の場合、選択するフィーチャーの絶対数を指定する。0 から 1 の間の float ならば、選択するフィーチャーの割合である。
    * バージョン0.24での変更点：分数にfloat値を追加。
* step: int or float, default=1
    * 1 以上の場合、step は各反復で削除する特徴量の（整数）個に対応する。(0.0, 1.0) 以内であれば、step は、各反復で削除するフィーチャーの割合 (切り捨て) に対応する。
* verbose: int, default=0
    * 出力の冗長性を制御する。

* importance_getter: str or callable, default=’auto’
    * auto' の場合、estimator の coef_ 属性と feature_importances_ 属性のどちらかを使って、特徴の重要度を得る。
    * attrgetterで実装された特徴量抽出のための属性名/パスを指定する文字列も受け付ける。例えば、TransformedTargetRegressor の場合は regressor_.coef_ を、clf という名前の最後のステップを持つ class:~sklearn.pipeline.Pipeline の場合は named_steps.clf.feature_importances_ を指定することができる。
    * callable の場合、デフォルトの特徴量ゲッターをオーバーライドします。callable は適合した推定量と共に渡され、各特徴の重要度を返す必要がある。