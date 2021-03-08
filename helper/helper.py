# -*- coding: utf-8 -*-

# import os
# path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
# os.chdir(path)
# %run helper.py

def getGeneralInformation(df):
    print(df.info)
    print()
    print(type(df))
    print()
    print(df.describe)
    print()
    print(df.dtypes)

def pre_work(df):
    num_var_nan=list()
    num_var_nonan=list()
    str_var_nan=list()
    str_var_nonan=list()
    for i in df.columns:
        if df[i].dtypes != 'object':
            if df[i].isna().sum()>0:
                num_var_nan.append(i)
            else:
                num_var_nonan.append(i)
        else:
            if df[i].isna().sum()>0:
                str_var_nan.append(i)
            else:
                str_var_nonan.append(i)
    return list(num_var_nan+ num_var_nonan+ str_var_nan+ str_var_nonan), num_var_nan, num_var_nonan, str_var_nan, str_var_nonan

### plotting
def plot_num_var(df,var):
    import matplotlib.pyplot as plt
    for i in var:
        # print(df[i].value_counts())
        if df[i].dtypes != 'object':
            nbr_nan=df[i].isna().sum()
            perc_nan=nbr_nan/len(df[i])
            plt.hist(df[i])
            plt.title("Numeric var: {}, cnt: {}, perc: {}".format(i,nbr_nan,round(perc_nan,2)))
            plt.show()
            plt.clf()


def plot_str_var(df,var):
    import matplotlib.pyplot as plt
    import seaborn as sns
    for i in var:
        if df[i].dtypes == 'object':
            nbr_nan=df[i].isna().sum()
            perc_nan=nbr_nan/len(df[i])
            sns.countplot(x=df[i].fillna('Missing'),order = df[i].fillna('Missing').value_counts().index)
            plt.title("String var: {}, cnt: {}, perc: {}".format(i,nbr_nan,round(perc_nan,2)))
            plt.show()
            plt.clf()


def get_violinplot_for_target(df,var,target):
    import seaborn as sns
    import matplotlib.pyplot as plt
    for i in var:
        sns.violinplot(x=df[target], y=df[i])
        plt.show()
        plt.clf()


def get_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="RdYlGn")
    plt.show()


def get_scatter_for_target(df,var,target):
    import matplotlib.pyplot as plt
    for i in var:
        plt.scatter(x=df[target], y=df[i])
        plt.title("{} vs {}".format(i,target))
        plt.show()
        plt.clf()


def impute_var(df,var,perc_drop,style):
    import numpy as np
    var_drop=[]
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    df[i].fillna(value=df[i].mean(),inplace=True)
                if style == 'median':
                    df[i].fillna(value=df[i].median(),inplace=True)
                if style == 'nan':
                    df[i].fillna(value=np.nan,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    df[i].fillna(value='missing',inplace=True)
            if style == 'mode':
                df[i].fillna(value=df[i].mode(dropna=True).values[0],inplace=True)
    return var_drop


def impute_var_v2(df,var,perc_drop,style):
    import numpy as np
    lst_var_drop=[]
    lst_impute=[]
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            lst_var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    impute_value=df[i].mean()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'median':
                    impute_value=df[i].median()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'nan':
                    impute_value=np.nan
                    df[i].fillna(value=impute_value,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    impute_value='missing'
                    df[i].fillna(value=impute_value,inplace=True)
            if style == 'mode':
                impute_value=df[i].mode(dropna=True).values[0]
                df[i].fillna(value=impute_value,inplace=True)
            lst_impute.append([i,impute_value])
    return lst_var_drop,lst_impute


def impute_var_v3(df,var,perc_drop,style):
    import numpy as np
    lst_var_drop=[]
    lst_impute=dict()
    #add drop na for full df
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            lst_var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    impute_value=df[i].mean()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'median':
                    impute_value=df[i].median()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'nan':
                    impute_value=np.nan
                    df[i].fillna(value=impute_value,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    impute_value='missing'
                    df[i].fillna(value=impute_value,inplace=True)
            if style == 'mode':
                impute_value=df[i].mode(dropna=True).values[0]
                df[i].fillna(value=impute_value,inplace=True)
            lst_impute[i]=impute_value
    return lst_var_drop,lst_impute


def impute_var_v4(df,var,perc_drop,style,value=None):
    import numpy as np
    lst_var_drop=[]
    lst_impute=dict()
    #add drop na for full df
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            lst_var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    impute_value=df[i].mean()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'median':
                    impute_value=df[i].median()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'nan':
                    impute_value=np.nan
                    df[i].fillna(value=impute_value,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    impute_value='missing'
                    df[i].fillna(value=impute_value,inplace=True)
            if style == 'mode':
                impute_value=df[i].mode(dropna=True).values[0]
                df[i].fillna(value=impute_value,inplace=True)
            elif style == 'value':
                impute_value=value
                df[i].fillna(value=impute_value,inplace=True)
            lst_impute[i]=impute_value
    return lst_var_drop,lst_impute


def low_corr(df,target,min_cor):
    cor=df.corr()
    drop_list_lowCor=cor[abs(cor[target])<=min_cor]
    return list(drop_list_lowCor.index)


def get_df_high_corr_target(df,target,min_cor):
    cor=df.corr()
    df_highCor=cor[abs(cor[target])>=0.5]
    lst_highCor=list(df_highCor.index)
    if target in lst_highCor:
        lst_highCor.remove(target)
    return df_highCor, lst_highCor


def merge_low_corr(df_ind,df_dep,target,min_cor):
    import pandas as pd
    df=pd.merge(left=df_ind,right=df_dep,how="inner",left_index=True,
    right_index=True)
    cor=df.corr()
    drop_list_lowCor=cor[abs(cor[target])<=min_cor]
    return list(drop_list_lowCor.index)


def same_value(df,var,max_perc_rep):
    drop_list_max_perc_rep=[]
    for i in var:
        if (df[i].value_counts().max()/len(df[i]))>=max_perc_rep:
            drop_list_max_perc_rep.append(i)
    return drop_list_max_perc_rep


def Standardize_values(df):
    from sklearn.preprocessing import StandardScaler
    SC = StandardScaler(with_mean=False)
    SC.fit(df)
    return SC


def Normalize_values(df):
    from sklearn.preprocessing import Normalizer
    transformer = Normalizer().fit(df)
    transformer.fit(df)
    return transformer


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def no_intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value not in lst2] 
    return lst3 


# def treat_str_var(df,var):
#     from sklearn.preprocessing import LabelEncoder # Converts cat data to numeric
#     le=LabelEncoder()
#     for i in var:
#         df[i]=le.fit_transform(df[i])


def tree_to_code(tree, feature_names):
    from sklearn.tree import _tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    

def get_tree_pic(df_X, df_Y, var):
    from sklearn.tree import DecisionTreeClassifier    
    import numpy as np
    from sklearn.tree import export_graphviz
    from six import StringIO
    from IPython.display import Image  
    import pydotplus

    if len(var)==1:
        split=np.array(df_X[var]).reshape(-1, 1)
    else:
        split=df_X[var]
    classifier = DecisionTreeClassifier(random_state = 0, max_depth=2)
    
    classifier.fit(split, df_Y)
    
    dot_data = StringIO()
    export_graphviz(classifier
                    ,out_file=dot_data
                    ,filled=True
                    ,rounded=True
                    ,special_characters=True
                    ,feature_names = var
                    ,proportion=True
                    ,rotate=True
                    ,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # graph.write_png('tree.png')
    Image(graph.create_png())
    return classifier, Image(graph.create_png())


def get_bins(df,var,nbr_bins):
    #pd.qcut(factors, 5).value_counts() #fixed volume
    #pd.cut(factors, 5).value_counts() #fixed intervalls e.g. 80/5=16
    import pandas as pd
    for i in var:
        df[i+'_bin_vol'] = pd.qcut(df[i], nbr_bins)
        df[i+'_bin_int'] = pd.cut(df[i], nbr_bins)
        print(df[i+'_bin_vol'].value_counts())
        print(df[i+'_bin_int'].value_counts())


def get_var_value_counts(df,var):
    lst_var_with_value_counts=[]
    for i in var:
        lst_var_with_value_counts.append([i, len(list(df[i].value_counts().index)),list(df[i].value_counts().index)])
    return lst_var_with_value_counts


# def create_flags(x):
#     import re
#     import pandas as pd
#     if pd.isnull(x):
#         return 3
#     elif re.search(r'[A-Za-z]',x) != None:
#         return 1
#     elif re.search(r'[0-9]',x) != None:
#         return 2
#     else:
#         return 4


def create_flags(x):
    import re
    import pandas as pd
    if pd.isnull(x):
        return 'M'
    elif re.search(r'[A-Za-z]',x) != None:
        return 'C'
    elif re.search(r'[0-9]',x) != None:
        return 'N'
    else:
        return 'O'


def create_log_var(df, num_var):
    #import seaborn as sns
    apply_value_log=[]
    from scipy.stats import skew
    skewed_feats = df[num_var].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    for i in skewed_feats.index:
        if skewed_feats[i]>=0.7:
            #   sns.displot(df[i])
            df['log_'+i]=np.log1p(df[i])
            # sns.displot(df['log_'+i])
            apply_value_log.append(i)
    return apply_value_log