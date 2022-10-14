# EDA
## 年推移
* アイスは夏に人気(分布には商品間で差がある)
* おでんは冬に人気
* おでん夏やってない。
* ホットスナック周期性ありそう
* デザートは週で周期性ありそう？
* ドリンクは売上の下限値がありそう
    * drink1~4は夏人気 (清涼飲料水？)
    * drink5~6は横ばい (コーヒーとか？)

![trend_per_year.png](../work/eda/trend_per_year.png)

## 週の周期性

週ごとに平均した値をヒートマップ化

![trend_per_week.png](../work/eda/trend_per_week.png)

週ごとに平均した値を標準化し、ヒートマップ化

![trend_std_per_week.png](../work/eda/trend_std_per_week.png)

## 日毎の周期性
日ごとに平均した値を標準化し、ヒートマップ化