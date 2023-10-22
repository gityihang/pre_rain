import torch

import pandas as pd
import torch.optim as optim

from model import ClaModel,RegModel

data = pd.read_csv('data/pre_20200901.csv')

data = data.sample(frac=1).reset_index(drop=True)  #打乱数据所在的行

#data = data.drop(data.columns[0], axis=1)

def IFRAIN(data,row_name):
    data['If_rain'] = 0 
    for index, row in data.iterrows():  
        if row[row_name] != 0:  
            data.at[index, 'If_rain'] = 1 
    # 保存结果到新的CSV文件  
    data.to_csv("new_data.csv", index=False)

def DeletRow(data, row_names):
    data.drop(columns=row_names, axis=1, inplace=True)

DeletRow(data, row_names=['Station_Name', 'Year', 'Mon', 'Day', 'Hour'])
data.to_csv("new_data.csv", index=False)


# inputs= data.drop(columns='PRE_1h').values

# outputs =   data.iloc[:,11].values

# inputs = torch.Tensor(inputs)

# outputs = torch.Tensor(outputs)




