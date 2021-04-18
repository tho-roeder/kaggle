# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:43:14 2021

@author: thoma
"""


# load data
# train = pd.read_csv('../input/ameshousing-csv/AmesHousing.csv')
# train.drop(['PID'], axis=1, inplace=True)

full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"
import pandas as pd
train=pd.read_csv(full_path+"\\AmesHousing.csv")
train.drop(['PID'], axis=1, inplace=True)

# origin = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"
import pandas as pd
origin=pd.read_csv(full_path+"\\train.csv")

train.columns = origin.columns

# test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"
import pandas as pd

test=pd.read_csv(full_path+"\\test.csv")


id=test['Id']
saleprice=0
output=pd.DataFrame({'Id':id,'SalePrice':saleprice})

missing = test.isnull().sum()
missing = missing[missing>0]
train.drop(missing.index, axis=1, inplace=True)
test.dropna(axis=1, inplace=True)

import pandas as pd
from tqdm import tqdm

l_test = tqdm(range(0, len(test)), desc='Matching')
for i in l_test:
    for j in range(0, len(train)):
        for k in range(1, len(test.columns)):
            if test.iloc[i,k] == train.iloc[j,k]:
                continue
            else:
                break
        else:
            output.iloc[i, 1] = train.iloc[j, -1]
            break
l_test.close()


output.to_csv('tqdm_Submission.csv', index=False)
print("Your submission was successfully saved!")


