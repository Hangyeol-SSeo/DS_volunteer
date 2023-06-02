# 0: 프로그램일련번호
# 1: 등록구분코드명
# 2: 상세유형코드명 - 
# 3: 활동시도코드명
# 4: 활동영역코드
# 5: 상세내용코드명 - 
# 6: 활동시군구코드명 - 
# 7: 인증시간내용
# 8: 회차
# 9: 상태코드명
# 10: 모집인원구분코드명
# 11: 모집인원수
# 12: 활동시작일시
# 13: 활동시작시간
# 14: 활동종료일시
# 15: 활동종료시간

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 결측치 처리하는 함수 / methodFill -> 'ffill', 'bfill', 'mode'
def fill_missing(df=None, column=None, methodFill='ffill', threshNum=2):
  if(methodFill == 'mode'):
    mode = df[column].mode()
    df[column] = df[column].fillna(str(mode[0]))
  else:
    df[column] = df[column].fillna(method=methodFill, limit=threshNum)
  return df

# Finding outlier data using the iqr
def get_outlier(df=None, column=None, weight=1.5):
  # target 값과 상관관계가 높은 열을 우선적으로 진행
  quantile_25 = np.percentile(df[column].values, 25)
  quantile_75 = np.percentile(df[column].values, 75)

  IQR = quantile_75 - quantile_25
  IQR_weight = IQR * weight
  
  lowest = quantile_25 - IQR_weight
  highest = quantile_75 + IQR_weight
  
  outlier_idx = df[column][ (df[column] < lowest) | (df[column] > highest) ].index
  return outlier_idx

# 데이터 불러오기
df = pd.read_csv("data/dataset_for_termproject.csv")
feature = ['PROGRM_SEQ_NO', 'REGIST_SDIV_CD_NM', 'DETAIL_TY_CD_NM', 
           'ACT_CTPRVN_CD_NM', 'ACT_RELM_CD', 'DETAIL_CN_CD_NM', 
           'ACT_SIGNGU_CD_NM', 'CRTFC_TIME_CN', 'TME', 
           'STATE_CD_NM', 'RCRIT_NMPR_SDIV_CD_NM', 'RCRIT_NMPR_CO', 
           'ACT_BEGIN_DT', 'ACT_BEGIN_TIME', 'ACT_END_DT', 'ACT_END_TIME']

# print(df.isna().sum())

# 함수 이용하여 결측치 제거
df = fill_missing(df=df, column=feature[5], methodFill='mode')
df = fill_missing(df=df, column=feature[2], methodFill='bfill', threshNum=45)
# print(df.isna().sum())

idx = df[df[feature[9]] == "삭제"].index
df.drop(idx , inplace=True)
df = df.replace('활동중', '활동완료')

numeric_end_time = df['ACT_BEGIN_TIME'].str.replace(':', '').str.isnumeric()
numeric_end_time = df['ACT_END_TIME'].str.replace(':', '').str.isnumeric()
df = df[numeric_end_time]

df['ACT_BEGIN_TIME'] = df['ACT_BEGIN_TIME'].str.replace(':', '').astype(int)
df['ACT_END_TIME'] = df['ACT_END_TIME'].str.replace(':', '').astype(int)

# 결측치 제거한 뒤에 시도명과 시군구 합치기.
df[feature[6]] = df[feature[6]].fillna('')
df[feature[3]] = df[feature[3]] + ' ' + df[feature[6]]
# print(df[feature[3]])

# 아웃라이어 솎아내기
oulier_idx = get_outlier(df=df, column=feature[8], weight=1.5)

# for i in oulier_idx:
#   print(df[feature[8]][i])

# 인코딩
encoder = LabelEncoder()

encoder.fit(df[feature[3]])
df[feature[3]] = encoder.transform(df[feature[3]])

# print(df['enc_city'].sort_values())
# print(df['enc_city'])

encoder.fit(df[feature[1]])
df[feature[1]] = encoder.transform(df[feature[1]])

encoder.fit(df[feature[2]])
df[feature[2]] = encoder.transform(df[feature[2]])

encoder.fit(df[feature[4]])
df[feature[4]] = encoder.transform(df[feature[4]])

encoder.fit(df[feature[5]])
df[feature[5]] = encoder.transform(df[feature[5]])

encoder.fit(df[feature[9]])
df[feature[9]] = encoder.transform(df[feature[9]])
encoder.fit(df['RCRIT_NMPR_SDIV_CD_NM'])
df["RCRIT_NMPR_SDIV_CD_NM"] = encoder.transform(df["RCRIT_NMPR_SDIV_CD_NM"])

x = df[[feature[1], feature[2], feature[3], feature[4], feature[5], feature[7], feature[8], feature[10], feature[11], feature[12], feature[13]]]
y = df[feature[9]]

#######################################################################################################################################################################################################

from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# 클러스터링에 사용할 feature 선택
numeric_feature = ['ACT_BEGIN_TIME', 'ACT_END_TIME']

out = ['ACT_BEGIN_TIME', 'ACT_END_TIME']
for feat in out:
  outlier = get_outlier(df=df, column=feat, weight=1.5)
  df = df.drop(labels=outlier, axis=0)

X = df[numeric_feature]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
scaled_df = pd.DataFrame(standard_scaler.fit_transform(X.iloc[:,0:4]), columns=X.iloc[:,0:4].columns) # scaled된 데이터

print(scaled_df)

#######################################################################################################################################################################################################

# ks = range(1,10)
# inertias = []

# for k in ks:
#     model = KMeans(n_clusters=k)
#     model.fit(scaled_df)
#     inertias.append(model.inertia_)

# # Plot ks vs inertias
# plt.figure(figsize=(4, 4))

# plt.plot(ks, inertias, '-o')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(ks)
# plt.show()

#######################################################################################################################################################################################################

# # create model and prediction
# # clust_model은 스케일링 전 fit과 동일하게 맞춤

# # K-means 알고리즘으로 클러스터링 수행
# kmeans = KMeans(n_clusters=5, n_init=10, random_state=42, max_iter=1000)
# kmeans.fit(scaled_df)

# centers_s = kmeans.cluster_centers_
# pred_s = kmeans.predict(scaled_df)

# print(pd.DataFrame(centers_s))
# print(pred_s[:10])

# clust_df = scaled_df.copy()
# clust_df['clust'] = pred_s
# print(clust_df.head())

# plt.figure(figsize=(20, 6))

# X = clust_df

# plt.subplot(131)
# sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], data=scaled_df, hue=kmeans.labels_, palette='coolwarm')

# plt.subplot(132)
# sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,2], data=scaled_df, hue=kmeans.labels_, palette='coolwarm')

# plt.subplot(133)
# sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,3], data=scaled_df, hue=kmeans.labels_, palette='coolwarm')

# plt.show()


#######################################################################################################################################################################################################

# import seaborn as sns
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import proj3d

# sns.pairplot(X)
# plt.show()

#######################################################################################################################################################################################################

# ks = range(1,10)
# inertias = []

# for k in ks:
#     model = KMeans(n_clusters=k, n_init=10)
#     model.fit(X)
#     inertias.append(model.inertia_)

# # Plot ks vs inertias
# plt.figure(figsize=(4, 4))

# plt.plot(ks, inertias, '-o')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(ks)
# plt.show()

#######################################################################################################################################################################################################

# K-means 알고리즘으로 클러스터링 수행
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42, max_iter=1000)
kmeans.fit(X)

# 클러스터링 결과 확인
cluster_labels = kmeans.labels_
df['Cluster'] = cluster_labels

# 클러스터별 데이터 개수 계산
cluster_counts = df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']

# 클러스터링 결과 데이터셋 출력
print(cluster_counts)
print(df.head())

# 값을 같은 것들끼리 그룹화하여 DataFrame 분할
grouped_dfs = []
for _, group_df in df.groupby("Cluster"):
    grouped_dfs.append(group_df)

# 분할된 DataFrame들 출력
for i, group_df in enumerate(grouped_dfs):
    samples = group_df.shape[0]
    complete = group_df[group_df['STATE_CD_NM'] == 0].shape[0]
    print(complete / samples)

# 결과 값을 변수에 저장
centers = kmeans.cluster_centers_ # 각 군집의 중심점
pred = kmeans.predict(X) # 각 예측군집

print(pd.DataFrame(centers))
print(pred[:10])

clust_df = X.copy()
clust_df['clust'] = pred
print(clust_df.head())

# scaling하지 않은 데이터를 학습하고 시각화하기

plt.figure(figsize=(20, 6))

A = clust_df
print(A)

plt.subplot(131)
sns.scatterplot(x=A.iloc[:,0], y=A.iloc[:,1], data=X, hue=kmeans.labels_, palette='coolwarm')
plt.scatter(centers[:,0], centers[:,1], c='black', alpha=0.8, s=150)

plt.show()

# 3차원으로 시각화하기

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# A = clust_df

# # 데이터 scatterplot
# ax.scatter(  A.iloc[:,3]
#            , A.iloc[:,1]
#            , A.iloc[:,2]
#            , c = A.clust
#            , s = 10
#            , cmap = "rainbow"
#            , alpha = 1
#           )

# # centroid scatterplot
# ax.scatter(centers[:,0],centers[:,1],centers[:,2] ,c='black', s=200, marker='*')
# ax.set_xlabel('ACT_END_TIME')
# ax.set_ylabel('RCRIT_NMPR_CO')
# ax.set_zlabel('ACT_BEGIN_TIME')
# plt.show()

# 시각화하여 표로 확인
plt.figure(figsize=(8, 6))
sns.barplot(x='Cluster', y='Count', data=cluster_counts)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Cluster Counts')
plt.show()

print('cluster의 평균값')
print(clust_df.groupby('clust').mean())

print('cluster의 최대값')
print(clust_df.groupby('clust').max())

print('cluster의 최소값')
print(clust_df.groupby('clust').min())

######################################################################################################################################################################################################

# # 값을 같은 것들끼리 그룹화하여 DataFrame 분할
# grouped_dfs = []
# for _, group_df in df.groupby(["DETAIL_CN_CD_NM", "DETAIL_TY_CD_NM", "REGIST_SDIV_CD_NM"]):
#     grouped_dfs.append(group_df)

# # 분할된 DataFrame들 출력
# for i, group_df in enumerate(grouped_dfs):
#     print(f"DataFrame {i+1}:")
#     print(group_df)
#     print()

#######################################################################################################################################################################################################

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Create a decision tree model
# model = DecisionTreeClassifier()

# # model training
# model.fit(x_train, y_train)

# # Predict with a trained model
# y_pred = model.predict(x_test)

# # Accuracy evaluation
# accuracy = accuracy_score(y_test, y_pred)

# # Prediction result and actual value output
# results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(results)
# print("Test Accuracy:", accuracy)

#######################################################################################################################################################################################################


# from sklearn.linear_model import LinearRegression

# mlr = LinearRegression()
# mlr.fit(x_train, y_train) 

# y_predict = mlr.predict(x_test)

# count=0
# cnt=0
# for i in y_test:
#   if (y_predict[count] > 0.4):
#     if (i == 1):
#       cnt += 1
#   count += 1

# print(cnt)
# print(count)
# # count = 0
# # for i in y_predict:
# #   if (i > 0.6):
# #     print(y_predict[count], end=" ")
# #   count += 1