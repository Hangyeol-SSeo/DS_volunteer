# 0: 프로그램일련번호 # 1: 등록구분코드명 # 2: 상세유형코드명 - 
# 3: 활동시도코드명 # 4: 활동영역코드 # 5: 상세내용코드명 - 
# 6: 인증시간내용 # 7: 회차 # 8: 모집인원구분코드명
# 9: 모집인원수 # 10: 활동시작시간 # 11: 활동종료시간

import preprocess_package
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("real_final_dataset.csv")
feature = ['PROGRM_SEQ_NO', 'REGIST_SDIV_CD_NM', 'DETAIL_TY_CD_NM', 
           'ACT_CTPRVN_CD_NM', 'ACT_RELM_CD', 'DETAIL_CN_CD_NM', 
           'CRTFC_TIME_CN', 'TME', 'RCRIT_NMPR_SDIV_CD_NM', 
           'RCRIT_NMPR_CO', 'ACT_BEGIN_TIME', 'ACT_END_TIME']

#######################################################
############### Data preprocessing step ###############
#######################################################

# Handling missing values using functions
df = preprocess_package.fill_missing(df=df, column=5, method_fill='mode')
df = preprocess_package.fill_missing(df=df, column=2, method_fill='bfill', thresh_num=45)

# Change data sample content
idx = df[df['STATE_CD_NM'] == "삭제"].index
df.drop(idx , inplace=True)
df = df.replace('활동중', '활동완료')

# Convert time String data to numeric
numeric_end_time = df[feature[10]].str.replace(':', '').str.isnumeric()
numeric_end_time = df[feature[11]].str.replace(':', '').str.isnumeric()
df = df[numeric_end_time]

df[feature[10]] = df[feature[10]].str.replace(':', '').astype(int)
df[feature[11]] = df[feature[11]].str.replace(':', '').astype(int)

# Merge 'ACT_CTPRVN_CD_NM' and 'ACT_SIGNGU_CD_NM' after removing missing values.
df['ACT_SIGNGU_CD_NM'] = df['ACT_SIGNGU_CD_NM'].fillna('')
df[feature[3]] = df[feature[3]] + ' ' + df['ACT_SIGNGU_CD_NM']

# encoding
encoder = LabelEncoder()

encoder.fit(df[feature[1]])
df[feature[1]] = encoder.transform(df[feature[1]])
encoder.fit(df[feature[2]])
df[feature[2]] = encoder.transform(df[feature[2]])
encoder.fit(df[feature[3]])
df[feature[3]] = encoder.transform(df[feature[3]])
encoder.fit(df[feature[4]])
df[feature[4]] = encoder.transform(df[feature[4]])
encoder.fit(df[feature[5]])
df[feature[5]] = encoder.transform(df[feature[5]])
encoder.fit(df['STATE_CD_NM'])
df['STATE_CD_NM'] = encoder.transform(df['STATE_CD_NM'])
encoder.fit(df[feature[8]])
df[feature[8]] = encoder.transform(df[feature[8]])

# Feature name to use for predictive model
x = df[[feature[1], feature[2], feature[3], feature[4], feature[5], feature[7], feature[8], feature[9], feature[10], feature[11]]]
y = df['STATE_CD_NM']

#######################################################################################################################################################################################################

########################################################
######## Finding the optimal number of clusters ########
########################################################

# Select features for clustering, remove outliers, paragraphs for scaling
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Select features to use for clustering
numeric_feature = [feature[10], feature[11]]

out = [10, 11]
for feat in out:
    out_lier = preprocess_package.get_outlier(df=df, column=[feat], weight=1.5)

X = df[numeric_feature]

# create model and prediction
# clust_model fits the same as fit before scaling
from sklearn.preprocessing import StandardScaler

# scaled data
standard_scaler = StandardScaler()
scaled_df = pd.DataFrame(standard_scaler.fit_transform(X.iloc[:,0:2]), columns=X.iloc[:,0:2].columns)

ks = range(1,10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(scaled_df)
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.figure(figsize=(4, 4))

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

######################################################################################################################################################################################################

######################################################
######## Clustering with scaling two feature #########
######################################################

# Select features for clustering, remove outliers, paragraphs for scaling
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Select features to use for clustering
numeric_feature = [feature[10], feature[11]]

out = [10, 11]
for feat in out:
    out_lier = preprocess_package.get_outlier(df=df, column=[feat], weight=1.5)

X = df[numeric_feature]

# create model and prediction
# clust_model fits the same as fit before scaling
from sklearn.preprocessing import StandardScaler

# scaled data
standard_scaler = StandardScaler()
scaled_df = pd.DataFrame(standard_scaler.fit_transform(X.iloc[:,0:2]), columns=X.iloc[:,0:2].columns)
data = df.reset_index(drop=True)
data[numeric_feature] = scaled_df

# Perform clustering with the KMeans algorithm
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42, max_iter=1000)
kmeans.fit(scaled_df)

# Check the clustering result
cluster_labels = kmeans.labels_
data['Cluster'] = cluster_labels

# Calculate the number of data per cluster
cluster_counts = data['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']

# Clustering result dataset output
print(cluster_counts)
print(data.head())

# Split the DataFrame by grouping the values into equals
grouped_dfs = []
for _, group_df in data.groupby("Cluster"):
    grouped_dfs.append(group_df)

# Calculation and output of success rate in partitioned DataFrames
for i, group_df in enumerate(grouped_dfs):
    samples = group_df.shape[0]
    complete = group_df[group_df['STATE_CD_NM'] == 0].shape[0]
    print(complete / samples)

# store the result in a variable
centers = kmeans.cluster_centers_ # the centroid of each cluster
pred = kmeans.predict(scaled_df) # each predicted cluster

print(pd.DataFrame(centers))
print(pred[:10])

clust_df = scaled_df.copy()
clust_df['clust'] = pred
print(clust_df.head())

# Visualize the result of learning scaled data
A = clust_df

# Visualize scaled results as a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=A.iloc[:,0], y=A.iloc[:,1], data=scaled_df, hue=kmeans.labels_, palette='coolwarm')
plt.scatter(centers[:,0], centers[:,1], c='black', alpha=0.8, s=150)
plt.show()

# Visualize the number of clusters and check them in a table
plt.figure(figsize=(8, 6))
sns.barplot(x='Cluster', y='Count', data=cluster_counts)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Cluster Counts')
plt.show()

# Output data for each cluster
print('cluster의 평균값')
print(clust_df.groupby('clust').mean())

print('cluster의 최대값')
print(clust_df.groupby('clust').max())

print('cluster의 최소값')
print(clust_df.groupby('clust').min())

#######################################################################################################################################################################################################

########################################################
######## Finding the optimal number of clusters ########
########################################################

# # Select features for clustering, remove outliers, paragraphs for scaling
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Select features to use for clustering
# numeric_feature = [feature[10], feature[11]]

# out = [10, 11]
# for feat in out:
#   df = preprocess_package.get_outlier(df=df, column=[feat], weight=1.5)

# X = df[numeric_feature]

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

# #######################################################################################################################################################################################################

#######################################################
####### Clustering with not scaling two feature #######
#######################################################

# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Select features to use for clustering
# numeric_feature = [feature[10], feature[11]]

# out = [10, 11]
# for feat in out:
#   df = preprocess_package.get_outlier(df=df, column=[feat], weight=1.5)

# X = df[numeric_feature]

# # Perform clustering with the KMeans algorithm
# kmeans = KMeans(n_clusters=3, n_init=10, random_state=42, max_iter=1000)
# kmeans.fit(X)

# # Check the clustering result
# cluster_labels = kmeans.labels_
# df['Cluster'] = cluster_labels

# # Calculate the number of data per cluster
# cluster_counts = df['Cluster'].value_counts().reset_index()
# cluster_counts.columns = ['Cluster', 'Count']

# # Clustering result dataset output
# print(cluster_counts)
# print(df.head())

# # Split the DataFrame by grouping the values into equals
# grouped_dfs = []
# for _, group_df in df.groupby("Cluster"):
#     grouped_dfs.append(group_df)

# # Calculation and output of success rate in partitioned DataFrames
# for i, group_df in enumerate(grouped_dfs):
#     samples = group_df.shape[0]
#     complete = group_df[group_df['STATE_CD_NM'] == 0].shape[0]
#     print(complete / samples)

# # store the result in a variable
# centers = kmeans.cluster_centers_ # the centroid of each cluster
# pred = kmeans.predict(X) # each predicted cluster

# print(pd.DataFrame(centers))
# print(pred[:10])

# clust_df = X.copy()
# clust_df['clust'] = pred
# print(clust_df.head())

# # Learning and visualizing non-scaling data
# A = clust_df

# # Visualize scaled results as a scatter plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=A.iloc[:,0], y=A.iloc[:,1], data=X, hue=kmeans.labels_, palette='coolwarm')
# plt.scatter(centers[:,0], centers[:,1], c='black', alpha=0.8, s=150)
# plt.show()

# # Visualize the number of clusters and check them in a table
# plt.figure(figsize=(8, 6))
# sns.barplot(x='Cluster', y='Count', data=cluster_counts)
# plt.xlabel('Cluster')
# plt.ylabel('Count')
# plt.title('Cluster Counts')
# plt.show()

# # Output data for each cluster
# print('cluster의 평균값')
# print(clust_df.groupby('clust').mean())

# print('cluster의 최대값')
# print(clust_df.groupby('clust').max())

# print('cluster의 최소값')
# print(clust_df.groupby('clust').min())