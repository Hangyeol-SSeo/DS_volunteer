import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from preprocess_package import fill_missing
from preprocess_package import get_outlier

pd.set_option('display.max_columns', None)
feature = ['PROGRM_SEQ_NO', 'REGIST_SDIV_CD_NM', 'DETAIL_TY_CD_NM',
           'ACT_CTPRVN_CD_NM', 'ACT_RELM_CD', 'DETAIL_CN_CD_NM',
           'ACT_SIGNGU_CD_NM', 'CRTFC_TIME_CN', 'TME',
           'STATE_CD_NM', 'RCRIT_NMPR_SDIV_CD_NM', 'RCRIT_NMPR_CO',
           'ACT_BEGIN_DT', 'ACT_BEGIN_TIME', 'ACT_END_DT', 'ACT_END_TIME']
os.chdir("/Users/ieunseob/gcu/3-1/DS/term_project/dataset")

# Load the dataset
data = pd.read_csv(
    "/Users/ieunseob/gcu/3-1/DS/term_project/dataset/real_final_dataset.csv",
    encoding='utf-8',
    low_memory=False)

# data preprocessing and analysis(Random Forest Algorithm)
# steps:
# 1. encode categorical data
# 2. process nan data
# 3. drop outliers
# 4. learn the model
# 5. predict the result

# 1. encode categorical data : label encoding
data = data[data[feature[9]] != '삭제']

data[feature[6]] = data[feature[6]].fillna('')
data[feature[3]] = data[feature[3]] + ' ' + data[feature[6]]

label_encoder = preprocessing.LabelEncoder()
encoded_column = label_encoder.fit_transform(data['REGIST_SDIV_CD_NM'])
data['REGIST_SDIV_CD_NM'] = encoded_column
encoded_column = label_encoder.fit_transform(data['DETAIL_TY_CD_NM'])
data['DETAIL_TY_CD_NM'] = encoded_column
encoded_column = label_encoder.fit_transform(data['ACT_CTPRVN_CD_NM'])
data['ACT_CTPRVN_CD_NM'] = encoded_column
encoded_column = label_encoder.fit_transform(data['ACT_RELM_CD'])
data['ACT_RELM_CD'] = encoded_column
encoded_column = label_encoder.fit_transform(data['DETAIL_CN_CD_NM'])
data['DETAIL_CN_CD_NM'] = encoded_column
encoded_column = label_encoder.fit_transform(data['RCRIT_NMPR_SDIV_CD_NM'])
data['RCRIT_NMPR_SDIV_CD_NM'] = encoded_column
label_encoder.fit(data[feature[3]])

numeric_end_time = data['ACT_BEGIN_TIME'].str.replace(':', '').str.isnumeric()
numeric_end_time = data['ACT_END_TIME'].str.replace(':', '').str.isnumeric()
data = data[numeric_end_time]

data['ACT_BEGIN_TIME'] = data['ACT_BEGIN_TIME'].str.replace(':', '').astype(int)
data['ACT_END_TIME'] = data['ACT_END_TIME'].str.replace(':', '').astype(int)

data[feature[3]] = label_encoder.transform(data[feature[3]])
data.drop(feature[6], axis=1, inplace=True)

# 2. process nan data
# split features data(feature 0~8, 10~13) and target(feature 9)
# feature 0~8, 10~13 : data
# feature 9 : target
target = data[feature[9]]
data.drop(feature[9], axis=1, inplace=True)

print(data)
print(target)
feature = ['PROGRM_SEQ_NO', 'REGIST_SDIV_CD_NM', 'DETAIL_TY_CD_NM',
           'ACT_CTPRVN_CD_NM', 'ACT_RELM_CD', 'DETAIL_CN_CD_NM',
           'CRTFC_TIME_CN', 'TME', 'RCRIT_NMPR_SDIV_CD_NM',
           'RCRIT_NMPR_CO', 'ACT_BEGIN_TIME', 'ACT_END_TIME']

proc_nan_feature = [1, 4]
proc_nan_limit = [20, 40]
proc_nan_method = ['bfill', 'ffill', 'mode']
feat_with_numeric = [5, 6, 8, 9, 10]
filled_data = pd.DataFrame()
for limit in proc_nan_limit:
    for method in proc_nan_method:
        filled_data = data.drop(feature[0], axis=1, inplace=False)
        for feat in proc_nan_feature:
            filled_data = fill_missing(df=filled_data, column=feat, method_fill=method, thresh_num=limit)

            # 3. drop outliers
            processed_data = get_outlier(df=filled_data, column=feat_with_numeric, weight=1.5)

            print("---------------------")
            print("nan limit : ", limit)
            print("nan method : ", method)

            # 4. learn the model
            # split the data into train and test data
            kf = KFold(n_splits=5, shuffle=True)
            kf.get_n_splits(processed_data)
            cv_accuracy = []
            for train_index, test_index in kf.split(processed_data):
                train_data, test_data = processed_data.iloc[train_index], processed_data.iloc[test_index]
                train_target, test_target = target.iloc[train_index], target.iloc[test_index]
                model = RandomForestClassifier(
                    n_estimators=10,
                    min_samples_split=5,
                    random_state=42
                    # max_depth=None
                )
                model.fit(train_data, train_target)

                # 5. predict the result with test data
                predicted = model.predict(test_data)

                # 6. cross validation
                # calculate the accuracy
                accuracy = accuracy_score(test_target, predicted)
                cv_accuracy.append(accuracy)

            # print the result(mean accuracy)
            print("accuracy : ", np.mean(cv_accuracy))
            print("---------------------")

