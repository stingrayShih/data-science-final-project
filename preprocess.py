import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import json

y = pd.read_csv('train_2.csv')

print('train2 before remove missing:',len(y['Page']))
y=y.dropna(axis=0, how='any')
print('train2 after remove missing:',len(y['Page']))


index=[]
names=[]

for i in range(len(y)):
    if '_zh' in y.iloc[i][0]:
        index.append(i)
        names.append(y.iloc[i][0])
        

dates=list(y.columns[1:])


split=[]
for _,d in enumerate(dates):
    if int(d.split('-')[2])==1:
        split.append(_)

dic={'names': names, 'dates': dates}


arr=[]
for i in index:
    print(i, y.iloc[i][0])   
    ts_decomposition = seasonal_decompose(y.iloc[i][1:], period=30)
    r=ts_decomposition.resid.to_numpy()
    mean=np.nanmean(r)
    std=np.nanstd(r)
    r=(r-mean)/std
    r=np.nan_to_num(r)
    arr.append(r)

    

arr=np.array(arr)
print(arr.shape)

for i in range(len(split)):
    month=dates[split[i]][:-3]
    if i==len(split)-1:
        print(arr[:,split[i]:].shape)
        np.save(f'zh_resid_{month}.npy', arr[:,split[i]:])
    else:
        print(arr[:,split[i]:split[i+1]].shape)
        np.save(f'zh_resid_{month}.npy', arr[:,split[i]:split[i+1]])


with open('train2_zh_date&name.json', 'w') as f:
    json.dump(dic, f)


