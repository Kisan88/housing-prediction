import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston        
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston=load_boston()
print(boston.DESCR)

#putting data into panda dataframe
boston.feature_names

data=boston.data
data.shape

data.describe()

data.info()

data.isnull()

sns.pairplot(data,height=2.5)

rows=2
cols=7
fig, ax=plt.subplots(nrows=rows,ncols=cols,figsize=(16,4))
col=data.columns
index=0

for i in range(rows):
    for j in range(cols):
        sns.distplot(data[col[index]],ax=ax[i][j])
        index=index+1
    
plt.tight_layout()  

corrmat=data.corr()
corrmat

fig, ax=plt.subplots(figsize= (18,10))
sns.heatmap(corrmat, annot = True, annot_kws={'size':12})

corrmat.index.values

features=pd.DataFrame(boston.data,columns=boston.feature_names)
features

# performing feature scaling on feature
standardscaler=preprocessing.StandardScaler()
features_scaled=standardscaler.fit_transform(features)
features_scaled

target=pd.DataFrame(boston.target,columns=['target'])
target

standardscaler=preprocessing.StandardScaler()
target_scaled=standardscaler.fit_transform(target)
target_scaled

#concatenating features and targets into one dataframe,setling axis=1 concatentes it column wise
df=pd.concat([features,target],axis=1)
df

#setting precisio to two places of decimal using round
df.describe().round(decimals=2)

#calculating correlation b/w every column on data
corr=df.corr('pearson')

#calculating correlation b/w every column on data
corr=df.corr('pearson')
#taking absolute value of correlation
corrs=[abs(corr[attr]['target'])for attr in list(features)]
#making list of pairs(corr,features)
l=list(zip(corrs,list(features)))
#sort the list of pairs in descending order according to correlation value
l.sort(key=lambda x:x[0],reverse=True)
#unzipping pairs to two lists
#zipping(*l) which takes input like[[a,b],[c,d]] and returns like [[a,c],[b,d]]
# corrs , labels=list(zip((*l)))
#plot correlations w.r.t the target variable as a bar graph
index=np.arange(len(labels))
plt.figure(figsize=(10,5))
plt.bar(index,corrs,width=0.7)
plt.xlabel('Attributes')
plt.ylabel('correlation with the target variables')
plt.xticks(index,labels)
plt.show()

X=df['LSTAT'].values
Y=df['target'].values
#before Normalization
Y[:5]

x_scaler=MinMaxScaler()
X=x_scaler.fit_transform(X.reshape(-1,1))
X=X[:-1]
y_scaler=MinMaxScaler()
Y=y_scaler.fit_transform(Y.reshape(-1,1))
Y=Y[:-1]

#after normalization
Y[:5]

#splitting the data into 80% train data & 20% test data
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)
xtrain=xtrain.reshape(-1,1)
ytrain=ytrain.reshape(-1,1)
xtest=xtest.reshape(-1,1)
ytest=ytest.reshape(-1,1)

#DEFINE REGTRESSION OBJECT
lm=LinearRegression()

#fitting the model
lm.fit(xtrain,ytrain)

#parameter of feature
lm.coef_

#making prediction
predictions=lm.predict(xtest)

#plotting
plt.scatter(ytest,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predictions')

#import evalution metrices
from sklearn import metrics
print('MSE:',metrics.mean_squared_error(ytest,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(ytest,predictions)))

p=pd.DataFrame(list(zip(xtest,ytest,predictions)),columns=['x','target_y','predictions'])
p

plt.scatter(xtest,ytest,color='b')
plt.plot(xtest,predictions,color='r')

#reshaping to change the shape required by scaler
predictions=predictions.reshape(-1,1)
xtest=ytest.reshape(-1,1)
ytest=ytest.reshape(-1,1)
xtest_scaled=x_scaler.inverse_transform(xtest)
ytest_scaled=y_scaler.inverse_transform(ytest)
predictions_scaled=y_scaler.inverse_transform(predictions)

#to remove extra dim
x_test_scaled=xtest_scaled[:-1]
y_test_scaled=ytest_scaled[:-1]
predictions_scaled=predictions_scaled[:-1]
p=pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,predictions_scaled)),columns=['x','target_y','predictions'])
p=p.round(decimals=2)
p