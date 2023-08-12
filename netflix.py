import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses,keras.metrics
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('Netflix Userbase.csv')
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())
print('---------------------------------------------------')
lab=LabelEncoder()
for i in data.columns.values:
    if len(data[i].value_counts())<= 4:
        print(data[i].value_counts().index.values)
        print(data[i].value_counts())

'''for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])'''

basic=data[data['Subscription Type']=='Basic']
standard=data[data['Subscription Type']=='Standard']
premium=data[data['Subscription Type']=='Premium']
print(premium)
print('------------------------')
print(standard)
print('----------------------------')
print(basic)
print('---------------------------------------')

print(data['Subscription Type'].value_counts().index.values)

for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])


'''for i in data.columns.values:
    if len(data[i].value_counts()) <= 5:
        for j in data[i].value_counts().index.values:
            print('------------------------------------------')
            print(f"The information about the column {i}")
            val=data[data[i]==j]
            for k in val.select_dtypes(include='object').columns.values:
                index = val[k].value_counts().index.values
                value = val[k].value_counts().values
                if (len(index) and len(value)) <= 10:
                    plt.pie(value, labels=index, autopct='%1.1f%%')
                    plt.title(f'the values and their counts related to {i} column')
                    plt.legend()
                    plt.show()

for i in data.columns.values:
        print(data[i].value_counts())
        print(data[i].value_counts().index)'''

'''plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.distplot(data[i], label=f"{i}", color='red')
        sn.distplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='red')
        sn.histplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''


y=[]
for i in data.columns.values:
    if len(data[i].value_counts()) <=4 and len(data[i].value_counts()) !=1:
        y.append(i)


for i in y:
    x=data.drop(i,axis='columns')
    y=data[i]
    print('------------------------------------')
    print('-------------------------------------------------')
    print(f'The Dependent Variables are {x.columns.values}')
    print(f' prediction regarding {i.upper()} column')
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(y_train.shape)

    lr = LogisticRegression(max_iter=200)
    lr.fit(x_train, y_train)
    print('The logistic regression: ', lr.score(x_test, y_test))

    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)
    print("the Xgb : ", xgb.score(x_test, y_test))

    lgb = LGBMClassifier()
    lgb.fit(x_train, y_train)
    print('The LGB', lgb.score(x_test, y_test))

    tree = DecisionTreeClassifier(criterion='gini', max_depth=1)
    tree.fit(x_train, y_train)
    print('Dtree ', tree.score(x_test, y_test))

    rforest = RandomForestClassifier(criterion='gini')
    rforest.fit(x_train, y_train)
    print('The random forest: ', rforest.score(x_test, y_test))

    adb = AdaBoostClassifier()
    adb.fit(x_train, y_train)
    print('the adb ', adb.score(x_test, y_test))

    grb = GradientBoostingClassifier()
    grb.fit(x_train, y_train)
    print('Gradient boosting ', grb.score(x_test, y_test))

    bag = BaggingClassifier()
    bag.fit(x_train, y_train)
    print('Bagging', bag.score(x_test, y_test))


X=data.drop(['Subscription Type'],axis=1)
Y=pd.get_dummies(data['Subscription Type'])
x_trin,x_tst,y_trin,y_tst=train_test_split(X,Y)

models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=Y.shape[1],activation=keras.activations.softmax))
models.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics='accuracy')
hist=models.fit(x_trin,y_trin,batch_size=20,epochs=10)
print(hist.history)
'''plt.plot(hist.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(hist.history['loss'], label='val_accuracy', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy')
plt.legend()
plt.show()'''