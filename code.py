import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import prince

#Loading the data
df = pd.read_csv("data.csv")
print(df.shape, "\n")
print(df.info(), "\n")
print(df.describe(), "\n")
#Seeing the Column Names and Unique Counts
for col in df:
    print(col + ' ' + str(df[col].nunique()))


#Cleaning the values with white spaces at the end
df=df.apply(lambda x: x.str.strip() if x.dtype=='object' else x)


#Finding null values and plotting the columns with null values
print(df.isna().mean())
df.isna().mean().plot(kind='barh')
df=df.loc[:,df.isna().mean() < .3]

#Dropping the null values
df=df.replace('NA', np.nan)
df=df.dropna()

#Finding the duplicate values and dropping
print(df.duplicated().sum())
#duplicate olanlari droplama
df=df.drop_duplicates()
df.info()


#Scatter Plot
fig, ax=plt.subplots(figsize=(12,8))
sns.scatterplot(x="Engine HP", y="city mpg", data=df)
plt.xticks(rotation=45)

#Count plot
fig,ax=plt.subplots(figsize=(12,8))
sns.countplot(x="Transmission Type", hue="Vehicle Style", data=df, ax=ax)
plt.xticks(rotation=45)

#Box Plot for Outliers
fig,ax=plt.subplots(figsize=(12,8))
sns.boxplot(x="Transmission Type", y="Engine HP", data=df)
plt.xticks(rotation=45)

#Visualizing the numeric columns by using each column in first 5 columns.
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
sns.pairplot(data=df, vars=numeric_cols[:5], hue="Number of Doors", palette='Set3')

#Finding the Correlation
df_num=df[numeric_cols]
df_corr=df_num.corr()
print(df_corr, "\n")
print(df_corr.mean(), "\n")
print(df_corr.abs().mean(), "\n")

#Correlation visualization
fig,ax= plt.subplots(figsize=(12,8))
sns.heatmap(df_corr, square=True, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20,220, n=200))
plt.xticks(rotation=45)

#Clustering(k-means)
print(df_num.columns.tolist())
X=StandardScaler().fit_transform(df_num)
kmeans=KMeans(n_clusters=5, init='random')
kmeans.fit(X)
pred=kmeans.predict(X)
np.unique(pred)

#Visualize Normalized Clustering
fig,ax=plt.subplots(figsize=(12,8))
plt.scatter(X[:,1], X[:,7], c=pred, cmap='viridis')
centers=kmeans.cluster_centers_
plt.scatter(centers[:,1], centers[:,7], c='grey', s=50)


#Principal Componant Analysis
pca = PCA(n_components=0.95)
pca.fit(X)
pcad=pca.transform(X)
print(pca.explained_variance_ratio_)

#PCA Visualization
fig,ax=plt.subplots(figsize=(12,8))
sns.scatterplot(pcad[:,0], pcad[:,1])


pca2 = prince.PCA(n_components=6, n_iter=3, rescale_with_mean=True,
                 rescale_with_std=True, copy=True, engine='auto')
pca2=pca2.fit(df_num)
pca2.explained_inertia_

ax=pca2.plot_row_coordinates(df_num, ax=None, figsize=(12,8),
x_component=0, y_component=1, labels=None,
color_labels=df['Transmission Type'],
ellipse_outline=False, ellipse_fill=True,
show_points=True)
