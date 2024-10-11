import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_csv("Car.csv")


#Cleaning 
print(df.isna().sum())


df=df.dropna()
df=df.reset_index(drop=True)


'''
Analyze each and every column wrt dependent
'''

'''
seats
'''
print(df['seats'].value_counts())




'''
torque
'''

print(df['torque'].dtype)

df=df.drop(columns="torque")
#Because of unstructure patterns we have to drop the column




'''
MAx power
'''
print(df['max_power'].dtype)

df['max_power']=df['max_power'].apply(lambda x:x.split()[0])

#df['max_power']=df['max_power'].astype("float32")
#Above line gave error find the noise value
l=[]
for i in range(len(df)):
    try: 
        float(df.iloc[i,-2])
    except:
        l.append(i)

df=df.drop(index=l).reset_index(drop=True)
df['max_power']=df['max_power'].astype("float32")




'''
Engine
'''
df['engine']=df['engine'].apply(lambda x:x.split()[0])
#df['engine']=df['engine'].astype("float32")

l=[]
for i in range(len(df)):
    try: 
        float(df.iloc[i,-3])
    except:
        l.append(i)

df=df.drop(index=l).reset_index(drop=True)
df['engine']=df['engine'].astype("float32")



'''
Mileage
'''

df['mileage']=df['mileage'].apply(lambda x:x.split()[0]).astype("float32")




'''
Owner
'''

print(df['owner'].value_counts())

df['owner']=df['owner'].replace({"Fifth":"Fourth & Above Owner"})

print(df['owner'].value_counts())

f=df['owner']=="Test Drive Car"
df=df.drop(index=df[f].index).reset_index(drop=True)
#Reson of dropping -->Test drive cars were considered as outliers in our data
#Thats why we drop it



'''
Transmission
'''
print(df["transmission"].value_counts())




'''
Seller Type
'''

print(df["seller_type"].value_counts())

for x in df["seller_type"].unique():
    f=df["seller_type"]==x
    plt.violinplot(df.loc[f,"selling_price"])
    plt.title(x)
    plt.ticklabel_format(style="plain")
    plt.show()


'''
fuel
'''

print(df["fuel"].value_counts())

for x in df["fuel"].unique():
    f=df["fuel"]==x
    plt.violinplot(df.loc[f,"selling_price"])
    plt.title(x)
    plt.ticklabel_format(style="plain")
    plt.show()


#Conclusion--> There is similar distribution of CNG & LPG
#AND similar distribution of Petrol & Diesel merge these categories

df["fuel"]=df["fuel"].replace({"CNG":0,"LPG":0,"Petrol":1,"Diesel":1})






print(df['name'].value_counts())
df['name']=df['name'].apply(lambda x: x.split()[0])
#The real impact is of brand on selling price thats why we convert 
#name column into brand column

brands=df['name'].unique()


#When number of categories is high then make groups of categories
#according to depedent variable

brand_selling_price=df.groupby("name")['selling_price'].mean()

brand_selling_price=brand_selling_price.sort_values(ascending=False)


#print(brand_selling_price.loc['Hyundai'])


def fxn(x):
    if x in brand_selling_price.iloc[:10]:
        return 0
    elif x in brand_selling_price.iloc[10:25]:
        return 1
    else:
        return 2



df['name']=df['name'].apply(fxn)





'''
Statistical analysis of columns
'''


categorical=df[['name','fuel','seller_type','transmission','owner','seats']]

numeric=df[['year','selling_price','km_driven','mileage','engine','max_power']]





from sklearn.preprocessing import LabelEncoder
encoder1=LabelEncoder()
categorical['owner']=encoder1.fit_transform(categorical['owner'])

encoder2=LabelEncoder()
categorical['transmission']=encoder2.fit_transform(categorical['transmission'])



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

onehotencode=OneHotEncoder(drop='first')
ct=ColumnTransformer([('encode',onehotencode,[2])],remainder='passthrough')
categorical=ct.fit_transform(categorical)



#sparse=False will not allow to form the sparse matrix after one hot encode 


'''
Feature selection
'''
#Four ways of feature selection

#1- Independent variable is numeric and dependent is also numeric
#2- Independent variable is numeric and dependent is category
#3- Independent variable is category and dependent is numeric
#4- Indepedent variable is category and dependent is category




#Methods for feature selection
#If One column is numeric and other is categorical  ---> ANOVA(Analysis of variance)


#If one column is numeric and other also numeric --> Pearson correlation method

#IF both columns are categorical  -->> Chi Sqaure Test

'''
Pearson correlation method
'''
#In this method we find the correlation coefficient(r)

corr=numeric.corr()
#It returns a correlation matrix


#To represent the correlation matrix in graphic way use heatmap
import seaborn
seaborn.heatmap(corr)
plt.show()

#HEatmap is used to reprent the correlation between different columns 




'''
ANOVA (Analysis of variance)
'''

#Anova is used to find out wether categories have same variance wrt
#other numeric(dependent) column or not

#In hypotheses testing we have two assumptions(cases)
#H0 -(Null hypothesis)  --> All categories have same variance
#H1 (Alternative hypothesis) --> Variance of all categories is different


#After testing we have to either reject or accept null hypothesis




#Steps to perform anova
#MAke assumptions (H0 and H1)
#Calculate Withtin sum of square and between sum of square
#Calculate degree of freedom of groups and each category
#Find f -value
#Take alpha value and confidence interval and find in graph wether to 
#accept or reject null hypothesis



#Within sum of square is square difference between each sample and their
#group mean

#Between sum of square is square differnce between each category's sample and grand 
#mean of all categories 





#F value represent if the test is statistical significant


from sklearn.feature_selection import SelectKBest,f_classif
#Select K best is used to sort the best columns according to test
#value passed as parameter
sk=SelectKBest(f_classif,k=7)
#k value represent the number of columns we want
#This value depends upon our domain knowledge

result=sk.fit_transform(categorical,numeric['selling_price'])
#In result ,there are top n columns with highest f score
print(sk.scores_)






'''
Outlier Detection
'''
#Outliers are those points which does not lie under the general pattern
#of a column
#Outliers are always checked in numeric
plt.scatter(numeric['km_driven'],numeric['selling_price'])
plt.show()


#Two cases:
    #When column is normally distributed
    #When column does not have any distribution
    
    
#Plot graph

for x in numeric.columns:
    plt.hist(numeric[x])
    plt.title(x)
    plt.show()


#To detect the outliers in gaussian distribution:
    #Q-outlier
    #Z-score
    #....
    
    
#To detect the outliers in non gaussian distribution:
    #DBSCAN->Density based spatial Clustering of application with noise 


def z_score(column):
    mean=column.mean()
    std=column.std()
    z=np.abs((column-mean)/std)
    return column[z>3]




outliers1=z_score(numeric['km_driven'])

outliers2=z_score(numeric['max_power'])
outliers3=z_score(numeric['mileage'])


#Task-->Drop the rows of outliers

f=~numeric['km_driven'].isin(z_score(numeric['km_driven']))
numeric=numeric[f]
categorical=categorical[f]
f=~numeric['max_power'].isin(z_score(numeric['max_power']))
numeric=numeric[f]
categorical=categorical[f]
f=~numeric['mileage'].isin(z_score(numeric['mileage']))
numeric=numeric[f]
categorical=categorical[f]



'''
DBSCAN
'''
#Outlier detection,Fraud detection, Clustering
#DBSCAN->Density based spatial Clustering of application with noise 

engine=numeric[['engine','selling_price']]

#Min max scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
engine=scaler.fit_transform(engine)
#To know the efficient value of epsilon draw k-distance graph
#K-distance graph tells us the overall average distances of nearest neighbor
#of each point

from sklearn.neighbors import NearestNeighbors

neighbor=NearestNeighbors(n_neighbors=10)


neighbor.fit(engine)
#Step 1-->Finding nearest neighbors

distance,index=neighbor.kneighbors(engine)
#In distance there will be array of shape(rows,neigbors)
#Each row of this array represent the distances of that point with its n 
#neighors
#Step 2-> Finding the distances

#Step 3--> Sort the distances
distance=np.sort(distance,axis=0)

#Step4 -->Extract out distances of neasrest neighbor

distance=distance[:,1]

#Step5--> MAke kdistance graph

plt.plot(distance)
plt.title("K-Distance Graph")
plt.show()
#For best epsilon choose value where graph start
# drastically increase


#Applyting DBSCAN algorithim
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.01,min_samples=10)
model=dbscan.fit(engine)

points=model.labels_

plt.scatter(numeric["engine"],
            numeric["selling_price"],
            c=points)
plt.show()



#Below filtering will keep only those 
#points whose respective labels(noise) is not 
#equal to -1

numeric=numeric[points!=-1]
categorical=categorical[points!=-1]



#Confine numeric and categorical
#Scaling ->Standard scling
#Build Regression model and find metrics









































