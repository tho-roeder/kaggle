# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 08:59:17 2021

@author: thoma
"""

import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
print(os.getcwd())
%run helper.py


def import_file(full_path):
    import pandas as pd
    df=pd.read_csv(full_path+"\\train.csv")
    return df


full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\titanic"
df=import_file(full_path)

all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df)   

