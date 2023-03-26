#https://www.codecademy.com/paths/fe-path-feature-engineering/tracks/fe-transforming-data-into-features/modules/fe-data-transformations-for-feature-analysis/articles/fe-encoding-categorical-variables
#https://pandas.pydata.org/pandas-docs/version/0.23/api.html#datetimelike-properties

def FeatureCenteringHist(df, var):
    import numpy as np
    import matplotlib.pyplot as plt 
    centered =df[var]-np.mean(df[var])
    print("The mean is {}".format(np.mean(df[var])))
    plt.title('Histogram '+ var)
    plt.hist(centered)
    plt.show()

def FeatureCentering(df, var):
    """
        the variable mean is set to zero
        :param df: dataframe 
        :param var: variable to be set to zero
        :return: no return
    """
    # how to apply 
    import numpy as np
    df[var+"_centered"] =df[var]-np.mean(df[var])
    print("The mean is {}".format(np.mean(df[var])))
    
def FeatureStandardizing(df,var):
    # how to apply --> mean and std need to be applied to test as learn from train
    import numpy as np
    df[var+"_standardized"]=(df[var]-np.mean(df[var]))/(np.std(df[var]))
    print("The mean is {}".format(np.mean(df[var])))
    print("The std is {}".format(np.std(df[var])))
    
def FeatureStandardizing_v2(df,var):
    # would be good to return the object to be able to transform test with the same
    from sklearn.preprocessing import StandardScaler 
    import numpy as np
    df[var+"_standardized"]=StandardScaler().fit_transform(np.array(df[var]).reshape(-1,1))
     
def FeatureMinMaxNormalization(df,var):
    # how to apply --> mean and std need to be applied to test as learn from train
    import numpy as np   
    df[var+"_MinMaxNorm"] = (df[var] - np.min(df[var])) / (np.max(df[var]) - np.min(df[var]))
    print("The min is {}".format(np.min(df[var])))
    print("The max is {}".format(np.max(df[var])))

def FeatureMinMaxNormalization_v2(df,var):
    # would be good to return the object to be able to transform test with the same
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    df[var+"_MinMaxNorm"] = MinMaxScaler().fit_transform(np.array(df[var]).reshape(-1,1))

def FeatureOrdinalEncoding(df,var,map_dict):
    # create dictionary of label:values in order e.g. {'Excellent':5, 'New':4, 'Like New':3, 'Good':2, 'Fair':1} or {True:1,False:0}
    df[var+'_Encoded'] = df[var].map(map_dict)

def FeatureOrdinalEncoding_v2(df,var,map_list):
    # map_list e.g. ['Excellent', 'New', 'Like New', 'Good', 'Fair']
    # would be good to return the object
    from sklearn.preprocessing import OrdinalEncoder
    df[var+'_OrdinalEncoded'] = OrdinalEncoder(categories=[map_list]).fit_transform(df[var].values.reshape(-1,1))
    
def FeatureLabelEncoding(df,var):
    df[var+'_LabelEncoded']=df[var].astype('category').cat.codes
    
def FeatureLabelEncoding_v2(df,var):
    from sklearn.preprocessing import LabelEncoder
    df[var+'_LabelEncoded']=LabelEncoder().fit_transform(df[var])

def FeatureOneHotEncoding(df,var):
    import pandas as pd
    return df.join(pd.get_dummies(df[var])) # why not working?

def FeatureBinaryEncoding(df,var):
    from category_encoders import BinaryEncoder
    return df.join(BinaryEncoder(cols = [var], drop_invariant = True).fit_transform(df[var])) # why not working?

def FeatureHashingEncoding(df,var,n_components=5):
    from category_encoders import HashingEncoder
    return df.join(HashingEncoder(cols=var, n_components=n_components).fit_transform(df[var]))

def FeatureTargetEncoder(df,var,var2):
    from category_encoders import TargetEncoder
    return df.join(TargetEncoder(cols = var).fit_transform(df[var], df[var2]))


# def FeatureDateEncoder()
# https://pandas.pydata.org/pandas-docs/version/0.23/api.html#datetimelike-properties
#     reviews['review_date']=reviews['review_date'].astype('datetime64[ns]')
#     reviews['review_date']=pd.to_datetime(reviews['review_date'])


# def 
# # reviews.reset_index(inplace=True,drop=True)
# reviews = reviews.set_index('clothing_id')
