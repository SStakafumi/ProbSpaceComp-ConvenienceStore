# Three Approaches to Encoding Time Information as Features for ML Models

あなたが新しいデータサイエンス・プロジェクトを始めたばかりだと想像してください。目標は、ターゲット変数であるYを予測するモデルを構築することです。あなたはすでにステークホルダー/データエンジニアからいくつかのデータを受け取り、徹底的なEDAを行い、目下の問題に関連すると思われるいくつかの変数を選択しました。そして、ついに最初のモデルを構築しました。スコアは許容範囲ですが、あなたはもっと良いものができると信じています。どうすればいいのでしょうか？

フォローアップの方法はたくさんあります。1つは、使用した機械学習モデルの複雑さを増すことである。あるいは、より意味のある機能を考え出すようにし、（少なくとも当面は）現在のモデルを使い続けることもできる。

多くのプロジェクトにおいて、企業のデータサイエンティストとKaggleのようなデータサイエンスコンテストの参加者の両方が、後者、つまりデータからより意味のある特徴を特定することが、最小限の努力でモデルの精度を最も向上させることができるという点で意見が一致しています。

つまり、モデルから特徴量に複雑さを移行させるのです。特徴はそれほど複雑である必要はありません。しかし、理想的には、ターゲット変数と強くかつ単純な関係を持つ特徴を見つけることである。

多くのデータサイエンス・プロジェクトは、時間の経過に関する何らかの情報を含んでいる。そして、これは時系列予測の問題に限定されるものではありません。例えば、従来の回帰や分類の課題でも、そのような特徴を見つけることがよくあります。この記事では、日付に関連する情報を使って意味のある特徴を作成する方法を調査しています。3つのアプローチを紹介するが、その前にいくつかの準備が必要である。

## Setup and data
この記事では、主に非常に有名なPythonパッケージを使用しますが、比較的知られていないものとして、scikit-learnの機能を拡張する多数の便利な機能を含むライブラリであるscikit-legoに依存します。必要なライブラリは以下のようにインポートします。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklego.preprocessing import RepeatingBasisFunction
```

シンプルにするために、私たちは自分でデータを生成します。この例では、人工的な時系列を扱います。まず、4暦年のインデックスを持つ空のDataFrameを作成します（pd.date_rangeを使用します）。次に、2つのカラムを作成します。

* day_nr - 時間の経過を表す数値インデックス
* day_of_year - 年の序数日

最後に、時系列そのものを作成する必要がある。そのために、2つの変換されたサインカーブといくつかのランダムなノイズを組み合わせます。データの生成に使用したコードは、scikit-legoのドキュメントに含まれるコードに基づきます。

```python
# for reproducibility
np.random.seed(42)

# generate the DataFrame with dates
range_of_dates = pd.date_range(start="2017-01-01",
                           	End="2020-12-30")
X = pd.DataFrame(index=range_of_dates)

# create a sequence of day numbers
X["day_nr"] = range(len(X))
X["day_of_year"] = X.index.day_of_year


# generate the components of the target
signal_1 = 3 + 4 * np.sin(X["day_nr"] / 365 * 2 * np.pi)
signal_2 = 3 * np.sin(X["day_nr"] / 365 * 4 * np.pi + 365/2)
noise = np.random.normal(0, 0.85, len(X))

# combine them to get the target series
y = signal_1 + signal_2 + noise

# plot
y.plot(figsize=(16,4), title="Generated time series")
```

そして、新しいDataFrameを作成し、そこに生成された時系列を格納します。このDataFrameは、特徴工学の異なるアプローチを用いたモデルの性能比較に使用されます。


```python
results_df = y.to_frame()
results_df.columns = ["actuals"]
```

## Creating time-related features
このセクションでは、時間に関連する特徴を生成するために考慮された3つのアプローチについて説明します。

すぐに飛び込む前に、評価のフレームワークを定義する必要がある。我々のシミュレーション・データは、4年の期間からのオブザベーションを含む。我々は、生成されたデータの最初の3年間をトレーニングセットとして使用し、4年目について評価を行う。評価指標としてMean Absolute Error (MAE)を使用します。

以下では、2つの集合を切り離すのに役立つ変数を定義します。

```python
TRAIN_END = 3 * 365
```

## Approach #1: dummy variables
まず、皆さんもある程度は知っているであろうものから始めます。時間に関係する情報を符号化する最も簡単な方法は、ダミー変数を使うことです（ワンホットエンコーディングとも呼ばれます）。例を見てみましょう。

```python
X_1 = pd.DataFrame(
	data=pd.get_dummies(X.index.month, drop_first=True, prefix="month")
)
X_1.index = X.index
X_1
```

まず、DatetimeIndexから月の情報（1～12の範囲の整数としてエンコードされている）を抽出した。そして、ダミー変数を作成するために pd.get_dummies 関数を使用しました。各列は、オブザベーション（行）が与えられた月から来るかどうかの情報を含んでいます。

お気づきかもしれませんが、レベルを1つ下げて、11列だけになりました。これは、線形モデルで作業するときに問題となる悪名高いダミー変数の罠（完全多重共線性）を回避するために行いました。

我々の例では、オブザベーションが記録された月を捕らえるために、ダミー変数のアプローチを使用しました。しかし、この同じアプローチは、DatetimeIndexから他の情報の範囲を示すために使用することができます。例えば、年の日/週/四半期、与えられた日が週末であるかどうかのフラグ、期間の最初の日/最後の日、その他多数です。pandas.pydata.orgにあるpandas documentation indexから、抽出できるすべての機能を含むリストを見つけることができます。

ボーナスヒント：これはこの簡単な演習の範囲外ですが、現実のシナリオでは、特別な日（国民の祝日、クリスマス、ブラックフライデーなどを考えてください）に関する情報を使って特徴を作成することもできます。holidaysは、国ごとの特別な日に関する過去と未来の情報を含む素晴らしいPythonライブラリです。

冒頭で述べたように、特徴量エンジニアリングの目標は、モデル側から特徴量側へ複雑性を移行させることです。そのため、最も単純なMLモデルの1つである線形回帰を用いて、作成したダミーだけを使ってどれだけ時系列にフィットできるかを確認します。

```python
model_1 = LinearRegression().fit(X_1.iloc[:TRAIN_END],
                             	y.iloc[:TRAIN_END])

results_df["model_1"] = model_1.predict(X_1)
results_df[["actuals", "model_1"]].plot(figsize=(16,4),
                                    	title="Fit using month dummies")
plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
```

フィットした線はすでに時系列によく沿っていることがわかりますが、少しギザギザしています（ステップ状） - ダミー特徴の不連続性に起因します。これは次の2つのアプローチで解決しようとするものです。

しかし、その前に、決定木のような非線形モデル（あるいはそのアンサンブル）を使う場合、月番や年号のような特徴をダミーとして明示的にエンコードしないことに言及する価値があるかもしれません。これらのモデルは、序数的な入力特徴と目標との間の非単調な関係を学習することができる。

## Approach #2: cyclical encoding with sine/cosine transformation

先に見たように、フィットした線は階段状になっている。これは、各ダミーが別々に扱われ、連続性がないためである。しかし、時間などの変数には、明らかに周期的な連続性が存在する。これはどういうことだろうか。

例えば、エネルギー消費量のデータを扱うとします。消費した月の情報を含めると、連続した2つの月の間にはより強いつながりがあることが分かります。この場合、12月と1月、1月と2月の結びつきは強いということになります。一方、1月と7月の結びつきはそれほど強くない。これは、時間に関する他の情報についても同じことが言えます。

では、この知識を特徴量工学に取り入れるにはどうすればよいのだろうか。三角関数の出番だ。以下のサイン/コサイン変換を使って、周期的な時間特徴を2つの特徴にエンコードすることができる。

```python
def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
```

以下のスニペットでは、最初のDataFrameをコピーし、月番号の列を追加し、sine/cosine変換を使用して月とday_of_year列の両方をエンコードしています。そして、両方の曲線のペアをプロットしています。

```python
X_2 = X.copy()
X_2["month"] = X_2.index.month

X_2["month_sin"] = sin_transformer(12).fit_transform(X_2)["month"]
X_2["month_cos"] = cos_transformer(12).fit_transform(X_2)["month"]

X_2["day_sin"] = sin_transformer(365).fit_transform(X_2)["day_of_year"]
X_2["day_cos"] = cos_transformer(365).fit_transform(X_2)["day_of_year"]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16,8))
X_2[["month_sin", "month_cos"]].plot(ax=ax[0])
X_2[["day_sin", "day_cos"]].plot(ax=ax[1])
plt.suptitle("Cyclical encoding with sine/cosine transformation");
```

図3に示した変換後のデータから、2つの知見を得ることができる。1つ目は、月単位でエンコードした場合は曲線が階段状になっているが、日単位でエンコードした場合は曲線がより滑らかになっていることが容易にわかること。2つ目は、なぜ1本の曲線ではなく2本の曲線を使わなければならないかもわかることだ。カーブの繰り返しの性質上、1年分のプロットにまっすぐな水平線を引くと、カーブを2箇所で横切ることになる。これでは、モデルが観測の時点を理解するのに十分ではないでしょう。しかし、2本のカーブでは、そのような問題はなく、ユーザーはすべてのタイムポイントを識別することができます。このことは、サイン/コサイン関数の値を散布図にプロットするとよくわかります。図4では、値が重なることなく、円形のパターンを見ることができる。

同じ線形回帰モデルを、日次頻度から来る新しく作成された特徴量だけを使ってフィットさせてみましょう。

```python
X_2_daily = X_2[["day_sin", "day_cos"]]

model_2 = LinearRegression().fit(X_2_daily.iloc[:TRAIN_END],
                             	y.iloc[:TRAIN_END])

results_df["model_2"] = model_2.predict(X_2_daily)
results_df[["actuals", "model_2"]].plot(figsize=(16,4),
                                    	title="Fit using sine/cosine features")
plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
```

図5から、このモデルはデータの大まかな傾向をつかむことができ、値が高い期間と低い期間を識別することができることがわかる。しかし、予測値の大きさについては精度が低く、一見するとダミー変数を用いた場合（図2）よりも適合性が悪いように見えます。

3 つ目の特徴量工学の手法を説明する前に、この手法には重大な欠点 があることを述べておく必要があり、それはツリーベースモデルを使 ったときに明らかになる。設計上、ツリーベースモデルは、その時々の単一の特徴に基づく分割を行う。そして、前にも述べたように、期間内の時点を適切に特定するためには、サイン/コサイン特徴を同時に考慮する必要があるのです。

## Approach #3: radial basis functions

最後の方法は、放射状基底関数を使う方法です。ラジアル基礎関数がどのようなものかについては、ここではあまり詳しく説明しませんが、このトピックについては、ここでもう少し詳しく読むことができます。基本的には、最初のアプローチで遭遇した問題、つまり時間特徴に連続性があることを再び解決したいのです。

RepeatingBasisFunctionクラスを提供する便利なscikit-legoライブラリを使用し、以下のパラメータを指定します。

* 作成したい基底関数の数（ここでは12を選択）。
* RBFのインデックスに使用するカラム。我々の場合、それは与えられたオブザベーションが何年の何月何日からのものであるかという情報を含むカラムです。
* 入力の範囲 - 我々のケースでは、範囲は1から365までです。
* 推定値のフィッティングに使用するDataFrameの残りの列をどうするか。「drop "は、作成されたRBFの特徴のみを残し、"passthrough "は、古い特徴と新しい特徴の両方を残します。

```python
rbf = RepeatingBasisFunction(n_periods=12,
                         	column="day_of_year",
                         	input_range=(1,365),
                         	remainder="drop")
rbf.fit(X)
X_3 = pd.DataFrame(index=X.index,
               	data=rbf.transform(X))

X_3.plot(subplots=True, figsize=(14, 8),
     	sharex=True, title="Radial Basis Functions",
     	legend=False)
```

図6は、日数を入力として作成した12個の放射状基底関数を示している。各曲線は、（その列を選んだので）その年のある日にどれだけ近いかという情報を含んでいる。例えば、最初の曲線は1月1日からの距離を測定しているので、毎年1月1日をピークに、その日から離れるにつれて対称的に減少する。

設計上、基底関数は入力範囲に等間隔で配置されている。RBFを月に似せたいので、12を選んだ。こうすることで、各関数は（月の長さが不等間隔であるため）月の初日までの距離をおおよそ示している。

先ほどのアプローチと同様に、12個のRBFの特徴量を用いて線形回帰モデルをあてはめましょう。

```python
model_3 = LinearRegression().fit(X_3.iloc[:TRAIN_END],
                             	y.iloc[:TRAIN_END])

results_df["model_3"] = model_3.predict(X_3)
results_df[["actuals", "model_3"]].plot(figsize=(16,4),
                                    	title="Fit using RBF features")
plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
```

図7は、RBF特徴を使用した場合、モデルが実データを正確に捉えることができることを示している。

ラジアル基底関数を使用する際に調整できる重要なパラメータは2つある。

* ラジアル基底関数の個数
* ベル曲線の形状 - これはRepeatingBasisFunctionのwidth引数で変更することができます。

これらのパラメータ値を調整する一つの方法は、与えられたデータセットに対して最適な値を特定するためにグリッドサーチを使用することである。

## Final comparison

以下のスニペットを実行すると、時間に関連する情報をエンコードするためのさまざまなアプローチの数値比較を生成することができます。

```python
results_df.plot(title="Comparison of fits using different time-based features",
            	figsize=(16,4),
            	color = ["c", "k", "b", "r"])
plt.axvline(date(2020, 1, 1), c="m", linestyle="--")
```

図8は、放射状基底関数が、検討したアプローチの中で最も適合度が高いことを示しています。図8は、放射状基底関数が検討されたアプローチの中で最も適合する結果となったことを示しています。サイン/コサイン機能は、モデルが主要なパターンをピックアップすることを可能にしますが、系列のダイナミクスを完全に捕らえるには十分ではありません。

以下のスニペットを用いて、トレーニングセットとテストセットの両方について、各モデルの平均絶対誤差を計算します。生成された系列はほぼ完全に循環しており、年ごとの違いはランダム成分だけなので、トレーニングセットとテストセットの間でスコアが非常に似ていることが予想されます。

当然ながら、現実の状況ではそうではなく、同じ期間の間でより多くの変動が発生することになります。しかし、そのような場合には、他の多くの特徴（例えば、トレンドや時間の経過の尺度）を用いて、それらの変化を説明することになります。

```python
score_list = []
for fit_col in ["model_1", "model_2", "model_3"]:
	scores = {
    	"model": fit_col,
    	"train_score": mean_absolute_error(
        	results_df.iloc[:TRAIN_END]["actuals"],
        	results_df.iloc[:TRAIN_END][fit_col]
    	),
    	"test_score": mean_absolute_error(
        	results_df.iloc[TRAIN_END:]["actuals"],
        	results_df.iloc[TRAIN_END:][fit_col]
    	)
	}
	score_list.append(scores)

scores_df = pd.DataFrame(score_list)
scores_df
```

前回と同様に、RBF特徴を用いたモデルが最も適合度が高く、サイン/コサイン特徴を用いたモデルは最も悪い結果となったことが分かります。また、トレーニングセットとテストセットのスコアの類似性についての我々の仮定も確認された。

# Takeaways
* 機械学習モデルの特徴量として、時間に関連する情報をエンコードする3つのアプローチを示した。
* 最も一般的なダミーエンコードの他に、時間の周期性をエンコードするのに適したアプローチもある。
* これらのアプローチを用いる場合、新たに生成される特徴量の形状は、時間間隔の粒度が大きく影響する。
* 放射状基底関数を用いると、使用する関数の数とベルカーブの幅を決めることができる。

この記事で使用したコードは、私のGitHubで見ることができます。万が一、フィードバックがあれば、Twitterで議論していただければと思います。