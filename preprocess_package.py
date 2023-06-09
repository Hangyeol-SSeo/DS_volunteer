import numpy as np
from sklearn import preprocessing


# function for scaling numerical data(target_column)
# m: min-max, z-score, robust
def normalizer(dataset, target_column, m):
    scaler = None
    if m == 'min-max':
        scaler = preprocessing.MinMaxScaler()
    elif m == 'z-score':
        scaler = preprocessing.StandardScaler()
    elif m == 'robust':
        scaler = preprocessing.RobustScaler()
    else:
        print("Invalid method")
        return None

    # scale data
    for c in target_column:
        dataset[feature[c]] = scaler.fit_transform(dataset[[feature[c]]])
    return dataset


# function for process nan data
# method_fill: ffill, bfill, mode
# thresh_num: threshold number for dropna
def fill_missing(df=None, column=None, method_fill='ffill', thresh_num=0):
    if method_fill == 'mode':
        mode = df[feature[column]].mode()
        df[feature[column]] = df[feature[column]].fillna(str(mode[0]))
    else:
        df[feature[column]] = df[feature[column]].fillna(method=method_fill, limit=thresh_num)
    return df.dropna(axis=0, how='any')


# function for drop outliers, using IQR
# weight: weight for iqr
def get_outlier(df=None, column=None, weight=1.5):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    for c in column:
        quantile_25 = np.percentile(df[feature[c]].values, 25)
        quantile_75 = np.percentile(df[feature[c]].values, 75)

        iqr = quantile_75 - quantile_25
        iqr_weight = iqr * weight

        lowest = quantile_25 - iqr_weight
        highest = quantile_75 + iqr_weight

        outlier_idx = df[feature[c]][(df[feature[c]] < lowest) | (df[feature[c]] > highest)].index
        df.drop(outlier_idx, axis=0)
    # drop outliers
    return df


feature = ['PROGRM_SEQ_NO', 'REGIST_SDIV_CD_NM', 'DETAIL_TY_CD_NM',
           'ACT_CTPRVN_CD_NM', 'ACT_RELM_CD', 'DETAIL_CN_CD_NM',
           'CRTFC_TIME_CN', 'TME', 'RCRIT_NMPR_SDIV_CD_NM',
           'RCRIT_NMPR_CO', 'ACT_BEGIN_TIME', 'ACT_END_TIME']
