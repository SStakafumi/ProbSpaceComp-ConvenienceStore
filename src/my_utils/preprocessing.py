import pandas as pd


def preprocess(df, mode='train', drop=False, year=2021):
    """method for preprocessing

    Args:
        df (DataFrame): DataFrame to be preprocessed
        mode (str, optional): 'train' or 'test'. Defaults to 'train'
        drop (bool, optional): delete 'date' column. Defaults to False
        year (int, optional): expected year. Defaults to 2021.

    Returns:
        DataFrame: preprocessed DataFrame
    """
    df_tmp = df.copy()

    df_tmp['time'] = pd.to_datetime(df_tmp.date, format='%m/%d')
    df_tmp['year'] = df_tmp['time'].dt.year
    df_tmp['month'] = df_tmp['time'].dt.month
    df_tmp['day'] = df_tmp['time'].dt.day

    if mode == 'train':
        df_tmp.loc[df_tmp['month'] > 3, 'year'] = year
        df_tmp.loc[df_tmp['month'] <= 3, 'year'] = year + 1
    else:
        df_tmp['year'] = year + 1
    df_tmp['time'] = pd.to_datetime(
        {'year': df_tmp.year, 'month': df_tmp.month, 'day': df_tmp.day})
    df_tmp['weekday'] = df_tmp['time'].dt.weekday
    df_tmp['day_of_year'] = df_tmp['time'].dt.day_of_year

    df_tmp = df_tmp.reindex(columns=[
        'id', 'date', 'time', 'year', 'month', 'day', 'weekday', 'day_of_year',
        'highest', 'lowest', 'rain', 'ice1', 'ice2', 'ice3', 'oden1', 'oden2',
        'oden3', 'oden4', 'hot1', 'hot2', 'hot3', 'dessert1', 'dessert2', 'dessert3',
        'dessert4', 'dessert5', 'drink1', 'drink2', 'drink3', 'drink4', 'drink5',
        'drink6', 'alcol1', 'alcol2', 'alcol3', 'snack1', 'snack2', 'snack3', 'bento1',
        'bento2', 'bento3', 'bento4', 'tild1', 'tild2', 'men1', 'men2', 'men3', 'men4', 'men5', 'men6'
    ])

    if drop:
        # drop 'date' and 'time' columns
        df_tmp = df_tmp.drop(['date'], axis=1)

    return df_tmp
