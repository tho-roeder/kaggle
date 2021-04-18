# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:11:04 2021

@author: thoma
"""

### Missing:
# https://www.kaggle.com/thoroeder/ensemble-stacked-regressors-top-3-92-acc/edit
# https://www.kaggle.com/thoroeder/data-science-predicting-housing-prices/edit
# https://www.kaggle.com/thoroeder/house-prices-regression-modelling-part-ii


#restrict pipeline to certain variables
# fix column transformer --> stacked?
# use pipeline
# build function to add basis impact while grid searching (or importance of Hyperparameter)

# from sklearn.metrics import roc_auc_score

# stack test train




####
# build stack

df = pd.concat([train, test])
df

def missing(df):
    miss = pd.DataFrame({"no_missing_values": df.isnull().sum(),
                         "missing_value_ratio": (df.isnull().sum() / df.shape[0]).round(4),
                         "missing_in_train": df[df.SalePrice.notnull()].isnull().sum(),
                         "missing_in_test": df[df.SalePrice.isnull()].isnull().sum()})
    return miss[miss.no_missing_values > 0].sort_values("no_missing_values", ascending=False)

missing(df)


### Feature engeniering

#semi manual:
df.loc[(df.GarageFinish.isnull()) & (df.GarageType.notnull()), "GarageFinish"] = "Fin"
df.loc[(df.GarageCars.isnull()) & (df.GarageType.notnull()), "GarageCars"] = 1
df.loc[(df.GarageQual.isnull()) & (df.GarageType.notnull()), "GarageQual"] = "TA"
df.loc[(df.GarageCond.isnull()) & (df.GarageType.notnull()), "GarageCond"] = "TA"


df.loc[(df.GarageYrBlt.isnull()) & (df.GarageType.notnull()), "GarageYrBlt"] = df.loc[(df.GarageYrBlt.isnull()) & (df.GarageType.notnull())].YearBuilt
df[(df.GarageType == "Detchd") & (df.YearBuilt < 1930) & (df.YearRemodAdd < 2000) & (df.YearRemodAdd > 1980) & (df.GarageCars == 1)].GarageArea.median()
df.loc[(df.GarageArea.isnull()) & (df.GarageType.notnull()), "GarageArea"] = 234
df.loc[(df.MasVnrArea.isnull()) & (df.MasVnrType == "None"), "MasVnrArea"] = 0

df.groupby("Neighborhood").MSZoning.value_counts()
df.loc[(df.MSZoning.isnull()) & (df.Neighborhood == "IDOTRR"), "MSZoning"] = "C (all)"
df.loc[(df.MSZoning.isnull()) & (df.Neighborhood == "Mitchel"), "MSZoning"] = "RL"


print(df.groupby(["OverallQual", "KitchenAbvGr"]).KitchenQual.value_counts())
df.loc[(df.KitchenQual.isnull()) & (df.OverallQual == 5 ) & (df.KitchenAbvGr == 1), "KitchenQual"] = "TA"


print(df.groupby(["Neighborhood", "SaleCondition"]).SaleType.value_counts())
df.loc[(df.SaleType.isnull()) & (df.Neighborhood == "Sawyer" ) & (df.SaleCondition == "Normal"), "SaleType"] = "WD"


df.Electrical.fillna("SBrkr", inplace=True)


print(df[(df.RoofMatl == "Tar&Grv")].Exterior1st.value_counts())
df.Exterior1st.fillna("Plywood", inplace=True)


print(df[(df.RoofMatl == "Tar&Grv")].Exterior2nd.value_counts())
df.Exterior2nd.fillna("Plywood", inplace=True)


df[(df.Neighborhood == "IDOTRR") &  (df.OverallQual < 5) & (df.YearRemodAdd < 1960) & (df.ExterQual == "Fa")].Functional.value_counts()
df.Functional.fillna("Mod", inplace = True)


df["LotFrontage"] = df["LotFrontage"].fillna(df.groupby(["Neighborhood", "LotShape", "LotConfig"])["LotFrontage"].transform("median"))
df["LotFrontage"] = df["LotFrontage"].fillna(df.groupby(["Neighborhood", "LotShape"])["LotFrontage"].transform("median"))
df["LotFrontage"] = df["LotFrontage"].fillna(df.groupby("Neighborhood")["LotFrontage"].transform("median"))


df["MSSubClass"] = df["MSSubClass"].astype("str")

df.loc[(df.GarageYrBlt == 2207), "GarageYrBlt"] = 2007

df.loc[(df.Exterior2nd == "CmentBd"), "Exterior2nd"] = "CemntBd"
df.loc[(df.Exterior2nd == "Wd Shng"), "Exterior2nd"] = "WdShing"
df.loc[(df.Exterior2nd == "Brk Cmn"), "Exterior2nd"] = "BrkComm"


# 3.1 For Categorical Features
# Converting some features to ordinal, extracting infos from features, combining bins that have same characteristics

dff["Older1945"] = dff["MSSubClass"].apply(lambda x: 1 if x in ["30", "70"] else 0)

dff["Newer1946"] = dff["MSSubClass"].apply(lambda x: 1 if x in ["20", "60", "120", "160"] else 0)

dff["AllStyles"] = dff["MSSubClass"].apply(lambda x: 1 if x in ["20", "90", "190"] else 0)

dff["AllAges"] = dff["MSSubClass"].apply(lambda x: 1 if x in ["40", "45", "50", "75", "90", "150", "190"] else 0)

dff["Pud"] = dff["MSSubClass"].apply(lambda x: 1 if x in ["120", "150", "160", "180"] else 0)

dff["Split"] = dff["MSSubClass"].apply(lambda x: 1 if x in ["80", "85""180"] else 0)

dff["MSSubClass"] = dff["MSSubClass"].apply(lambda x: "180" if x == "150" else x)




dff["MSZoning"] = dff["MSZoning"].apply(lambda x: "R" if x.startswith("R") else x)




dff["North"] = dff["Neighborhood"].apply(lambda x: 1 if x in ["Blmngtn", "BrDale", "ClearCr", "Gilbert",  "Names", "NoRidge", 
                                                              "NPkVill", "NWAmes", "NoRidge", "NridgHt", "Sawyer", "Somerst", 
                                                              "StoneBr", "Veenker", "NridgHt"] else 0)

dff["South"] = dff["Neighborhood"].apply(lambda x: 1 if x in ["Blueste", "Edwards", "Mitchel", "MeadowV", 
                                                              "SWISU", "IDOTRR", "Timber"] else 0)

dff["Downtown"] = dff["Neighborhood"].apply(lambda x: 1 if x in ["BrkSide", "Crawfor", "OldTown", "CollgCr"] else 0)

dff["East"] = dff["Neighborhood"].apply(lambda x: 1 if x in ["IDOTRR", "Mitchel"] else 0)

dff["West"] = dff["Neighborhood"].apply(lambda x: 1 if x in ["Edwards", "NWAmes", "SWISU", "Sawyer", "SawyerW"] else 0)


dff.loc[(dff["Condition1"] == "Feedr") | (dff["Condition2"] == "Feedr"), "StreetDegree"] = 1
dff.loc[(dff["Condition1"] == "Artery") | (dff["Condition2"] == "Artery"), "StreetDegree"] = 2
dff["StreetDegree"].fillna(0, inplace = True)

dff.loc[(dff["Condition1"].isin(["RRNn", "RRNe"])) | (dff["Condition2"].isin(["RRNn", "RRNe"])), "RailroadDegree"] = 1
dff.loc[(dff["Condition1"].isin(["RRAn", "RRAe"])) | (dff["Condition2"].isin(["RRAn", "RRAe"])), "RailroadDegree"] = 2
dff["RailroadDegree"].fillna(0, inplace = True)

dff.loc[(dff["Condition1"] == "PosN") | (dff["Condition2"] == "PosN"), "OffsiteFeature"] = 1
dff.loc[(dff["Condition1"] == "PosA") | (dff["Condition2"] == "PosA"), "OffsiteFeature"] = 2
dff["OffsiteFeature"].fillna(0, inplace = True)

dff["Norm1"] = dff["Condition1"].apply(lambda x: 1 if x == "Norm" else 0)
dff["Norm2"] = dff["Condition2"].apply(lambda x: 1 if x == "Norm" else 0)
dff["Norm"] = dff["Norm1"] + dff["Norm2"]
dff.drop(["Norm1", "Norm2"], axis = 1, inplace = True)

dff["BldgType"] = dff["BldgType"].apply(lambda x: "2Fam" if x in ["2fmCon", "Duplex"] else x)

dff["SaleType"] = dff["SaleType"].apply(lambda x: "WD" if x.endswith("WD") else x)
dff["SaleType"] = dff["SaleType"].apply(lambda x: "Contract" if x.startswith("Con") else x)
dff["SaleType"] = dff["SaleType"].apply(lambda x: "Oth" if x == "COD" else x)

dff["SaleCondition"] = dff["SaleCondition"].apply(lambda x: "Abnormal_Adjland" if x in ["Abnorml", "AdjLand"] else x)
dff["SaleCondition"] = dff["SaleCondition"].apply(lambda x: "Alloca_Family" if x in ["Alloca", "Family"] else x)
dff["SaleCondition"] = dff["SaleCondition"].apply(lambda x: "Other" if x in ["Abnormal_Adjland", "Alloca_Family"] else x)

dff["GarageType"] = dff["GarageType"].apply(lambda x: "Carport_None" if x in ["CarPort", "None"] else x)
dff["GarageType"] = dff["GarageType"].apply(lambda x: "Basement_2Types" if x in ["Basment", "2Types"] else x)

dff["LotConfig"] = dff["LotConfig"].apply(lambda x: "CulDSac_FR3" if x in ["CulDSac", "FR3"] else x)

dff["RoofStyle"] = dff["RoofStyle"].apply(lambda x: "Other" if x not in ["Gable"] else x)
dff["RoofMatl"] = dff["RoofMatl"].apply(lambda x: "Other" if x != "CompShg" else x)
dff["MasVnrType"] = dff["MasVnrType"].apply(lambda x: "None_BrkCmn" if x in ["None", "BrkCmn"] else x)

dff["Foundation"] = dff["Foundation"].apply(lambda x: "BrkTil_Stone" if x in ["BrkTil", "Stone"] else x)
dff["Foundation"] = dff["Foundation"].apply(lambda x: "BrkTil_Stone_Slab" if x in ["BrkTil_Stone", "Slab"] else x)
dff["Foundation"] = dff["Foundation"].apply(lambda x: "PConc_Wood" if x in ["PConc", "Wood"] else x)

dff["Heating"] = dff["Heating"].apply(lambda x: "Other" if x != "GasA" else x)


# 3.2 For Numerical Features
# Creating features with using feature interactions, creating binary features, new features with using ordinal ones

dff["FrontageRatio"] = (dff["LotFrontage"] / dff["LotArea"])
dff["HQFloor"] = dff["1stFlrSF"] + dff["2ndFlrSF"]
dff["FloorAreaRatio"] = dff["GrLivArea"] / dff["LotArea"]

dff["TotalArea"] = dff["TotalBsmtSF"] + dff["GrLivArea"]
dff["TotalPorch"] = dff["WoodDeckSF"] + dff["OpenPorchSF"] + dff["EnclosedPorch"] + dff["3SsnPorch"] + dff["ScreenPorch"]

dff["WeightedBsmtFinSF1"] = dff["BsmtFinSF1"] * dff["BsmtFinType1"]
dff["WeightedBsmtFinSF2"] = dff["BsmtFinSF2"] * dff["BsmtFinType2"]
dff["WeightedTotalBasement"] =  dff["WeightedBsmtFinSF1"] + dff["BsmtFinSF2"] * dff["BsmtFinType2"] +  dff["BsmtUnfSF"]

dff["TotalFullBath"] = dff["BsmtFullBath"] + dff["FullBath"]
dff["TotalHalfBath"] = dff["BsmtHalfBath"] + dff["HalfBath"]

dff["TotalBsmtBath"] = dff["BsmtFullBath"] + 0.5 * dff["BsmtHalfBath"]
dff["TotalBath"] = dff["TotalFullBath"] + 0.5 * (dff["BsmtHalfBath"] + dff["HalfBath"]) + dff["BsmtFullBath"] + 0.5 * dff["BsmtHalfBath"]

dff["HasPool"] = dff["PoolArea"].apply(lambda x: 0 if x == 0 else 1)
dff["Has2ndFlr"] = dff["2ndFlrSF"].apply(lambda x: 0 if x == 0 else 1)
dff["HasBsmt"] = dff["TotalBsmtSF"].apply(lambda x: 0 if x == 0 else 1)
dff["HasFireplace"] = dff["Fireplaces"].apply(lambda x: 0 if x == 0 else 1)
dff["HasGarage"] = dff["GarageArea"].apply(lambda x: 0 if x == 0 else 1)
dff["HasLowQual"] = dff["LowQualFinSF"].apply(lambda x: 0 if x == 0 else 1)
dff["HasPorch"] = dff["TotalPorch"].apply(lambda x: 0 if x == 0 else 1)
dff["HasMiscVal"] = dff["MiscVal"].apply(lambda x: 0 if x == 0 else 1)
dff["HasWoodDeck"] = dff["WoodDeckSF"].apply(lambda x: 0 if x == 0 else 1)
dff["HasOpenPorch"] = dff["OpenPorchSF"].apply(lambda x: 0 if x == 0 else 1)
dff["HasEnclosedPorch"] = dff["EnclosedPorch"].apply(lambda x: 0 if x == 0 else 1)
dff["Has3SsnPorch"] = dff["3SsnPorch"].apply(lambda x: 0 if x == 0 else 1)
dff["HasScreenPorch"] = dff["ScreenPorch"].apply(lambda x: 0 if x == 0 else 1)

dff["TotalPorchType"] = dff["HasWoodDeck"] + dff["HasOpenPorch"] + dff["HasEnclosedPorch"] + dff["Has3SsnPorch"] + dff["HasScreenPorch"]
dff["TotalPorchType"] = dff["TotalPorchType"].apply(lambda x: 3 if x >=3 else x)


dff["RestorationAge"] = dff["YearRemodAdd"] - dff["YearBuilt"]
dff["RestorationAge"] = dff["RestorationAge"].apply(lambda x: 0 if x < 0 else x)
dff["HasRestoration"] = dff["RestorationAge"].apply(lambda x: 0 if x == 0 else 1)

dff["YearAfterRestoration"] = dff["YrSold"] - dff["YearRemodAdd"]
dff["YearAfterRestoration"] = dff["YearAfterRestoration"].apply(lambda x: 0 if x < 0 else x)

dff["BuildAge"] = dff["YrSold"] - dff["YearBuilt"]
dff["BuildAge"] = dff["BuildAge"].apply(lambda x: 0 if x < 0 else x)
dff["IsNewHouse"] = dff["BuildAge"].apply(lambda x: 1 if x == 0 else 0)

def year_map(year):
    # 1: GildedAge, 2: ProgressiveEra, 3: WorldWar1, 4: RoaringTwenties, 5: GreatDepression, 
    # 6: WorlWar2, 7: Post-warEra, 8: CivilRightsEra, 9: ReaganEra, 10: Post-ColdWarEra, 11: ModernEra
    year = 1 if year <= 1895 else\
    (2 if year <= 1916 else\
     (3 if year <= 1919 else\
      (4 if year <= 1929 else\
       (5 if year <= 1941 else\
        (6 if year <= 1945 else\
         (7 if year <= 1964 else\
          (8 if year <= 1980 else\
           (9 if year <= 1991 else\
            (10 if year < 2008 else 11))))))))) 
    
    return year

dff["YearBuilt_bins"] = dff["YearBuilt"].apply(lambda year: year_map(year))
dff["YearRemodAdd_bins"] = dff["YearRemodAdd"].apply(lambda year: year_map(year))
dff["GarageYrBlt_bins"] = dff["GarageYrBlt"].apply(lambda year: year_map(year))

dff["YrSold"] = dff["YrSold"].astype(str)
dff["MoSold"] = dff["MoSold"].astype(str)
dff["Season"] = dff["MoSold"].apply(lambda x: "Winter" if x in ["12", "1", "2"] else\
                                   ("Spring" if x in ["3", "4", "5"] else\
                                   ("Summer" if x in ["6", "7", "8"] else "Fall")))
    
    
    
    
    
dff["OverallValue"] = dff["OverallQual"] * dff["OverallCond"]
dff["ExterValue"] = dff["ExterQual"] * dff["ExterCond"]
dff["BsmtValue"] = ((dff["BsmtQual"] + dff["BsmtFinType1"] + dff["BsmtFinType2"]) * dff["BsmtCond"]) / 2
dff["KitchenValue"] = dff["KitchenAbvGr"] * dff["KitchenQual"]
dff["FireplaceValue"] = dff["Fireplaces"] * dff["FireplaceQu"]
dff["GarageValue"] = dff["GarageQual"] * dff["GarageCond"]

dff["TotalValue"] = dff["OverallValue"] + dff["ExterValue"] + dff["BsmtValue"] + dff["KitchenValue"] + dff["FireplaceValue"] + dff["GarageValue"] +\
dff["HeatingQC"] + dff["Utilities"] + dff["Electrical"] - dff["Functional"]  + dff["PoolQC"]

dff["TotalQual"] = dff["OverallQual"] + dff["ExterQual"] + dff["BsmtQual"] + dff["KitchenQual"] + dff["FireplaceQu"] + dff["GarageQual"] +\
dff["HeatingQC"] + dff["PoolQC"]

dff["TotalCond"] = dff["OverallCond"] + dff["ExterCond"] + dff["BsmtCond"] + dff["GarageCond"]
dff["TotalQualCond"] = dff["TotalQual"] + dff["TotalCond"]



dff["BsmtSFxValue"] = dff["TotalBsmtSF"] * dff["BsmtValue"]
dff["BsmtSFxQual"] = dff["TotalBsmtSF"] * dff["BsmtQual"]

dff["TotalAreaXOverallValue"] = dff["TotalArea"] * dff["OverallValue"]
dff["TotalAreaXOverallQual"] = dff["TotalArea"] * dff["OverallQual"]

dff["GarageAreaXGarageValue"] = dff["GarageArea"] * dff["GarageValue"]
dff["GarageAreaXGarageQual"] = dff["GarageArea"] * dff["GarageQual"]



# 3.3 For Ordinal Features
# Combining bins, it will help us to get stronger correlations. It is useful espeically for linear models.

dff2["LotShape"] = dff2["LotShape"].apply(lambda x: 1 if x in [1, 2] else (2 if x == 3 else 3))
dff2["LandSlope"] = dff2["LandSlope"].apply(lambda x: 1 if x in [1, 2] else (2 if x == 3 else 3))
dff2["OverallCond"] = dff2["OverallCond"].apply(lambda x: 1 if x in [1, 2, 3] else x-1)
dff2["OverallQual"] = dff2["OverallQual"].apply(lambda x: 1 if x in [1, 2] else x-1)
dff2["ExterCond"] = dff2["ExterCond"].apply(lambda x: 1 if x in [1, 2] else (2 if x == 3 else 3))
dff2["BsmtQual"] = dff2["BsmtQual"].apply(lambda x: 0 if x in [0, 1, 2] else (1 if x == 3 else (2 if x == 4 else 3)))
dff2["BsmtCond"] = dff2["BsmtCond"].apply(lambda x: 0 if x in [0, 1, 2] else (1 if x == 3 else 2))
dff2["BsmtFinType1"] = dff2["BsmtFinType1"].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else (2 if x == 6 else x))
dff2["BsmtFinType2"] = dff2["BsmtFinType2"].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else (2 if x == 6 else x))
dff2["HeatingQC"] = dff2["HeatingQC"].apply(lambda x: 1 if x in [1, 2] else (2 if x in [3, 4] else 3))
dff2["Electrical"] = dff2["Electrical"].apply(lambda x: 1 if x in [1, 2] else x-3)
dff2["BsmtFullBath"] = dff2["BsmtFullBath"].apply(lambda x: 2 if x >= 2 else x)
dff2["FullBath"] = dff2["FullBath"].apply(lambda x: 1 if x <= 1 else (3 if x >= 3 else x))
dff2["HalfBath"] = dff2["HalfBath"].apply(lambda x: 1 if x >= 1 else 0)
dff2["BedroomAbvGr"] = dff2["BedroomAbvGr"].apply(lambda x: 1 if x <=1 else (5 if x >= 5 else x))
dff2["KitchenAbvGr"] = dff2["KitchenAbvGr"].apply(lambda x: 1 if x <= 1 else (2 if x >= 2 else x))
dff2["TotRmsAbvGrd"] = dff2["TotRmsAbvGrd"].apply(lambda x: 3 if x <= 4 else (10 if x >= 11 else x-1))
dff2["Functional"] = dff2["Functional"].apply(lambda x: 1 if x == 1 else 2)
dff2["Fireplaces"] = dff2["Fireplaces"].apply(lambda x: 2 if x >= 2 else x)
dff2["GarageCars"] = dff2["GarageCars"].apply(lambda x: 3 if x >= 3 else x)
dff2["GarageQual"] = dff2["GarageQual"].apply(lambda x: 1 if x <= 2 else (2 if x == 3 else 3))
dff2["GarageCond"] = dff2["GarageCond"].apply(lambda x: 1 if x <= 2 else 2)
dff2["Fence"] = dff2["Fence"].apply(lambda x: 1 if x in [1, 3] else x)




dff3["Exterior"] = np.where((dff3["Exterior1st"] != dff3["Exterior2nd"]), "Mixed", dff3["Exterior1st"])
dff3["No2ndExt"] = dff3["Exterior"].apply(lambda x: 0 if x == "Mixed" else 1)




lotshape = {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}
landcontour = {"Low":1, "HLS": 2, "Bnk":3, "Lvl": 4}
utilities = {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}
landslope = {"Sev": 1, "Mod": 2, "Gtl": 3}

general = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

bsmtexposure = {"None": 0, "No": 0, "Mn": 1, "Av": 2, "Gd": 3}
bsmtfintype = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
electrical = {"Mix": 1, "FuseP": 2, "FuseF": 3, "FuseA": 4, "SBrkr": 5}
functional = {"Typ": 1, "Min1": 2, "Min2": 3, "Mod": 4, "Maj1": 5, "Maj2": 6, "Sev": 7, "Sal": 8}
garagefinish = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
fence = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}

dff.replace({"LotShape": lotshape, "LandContour": landcontour, "Utilities": utilities, "LandSlope": landslope, 
             "BsmtExposure": bsmtexposure, "BsmtFinType1": bsmtfintype, "BsmtFinType2":bsmtfintype, "Electrical": electrical, 
             "Functional": functional, "GarageFinish": garagefinish, "Fence": fence}, 
             inplace = True)

for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", 
            "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]:
    dff[col] = dff[col].replace(general)
    
    
# 3. Comprehensive Eda & Feature Engineering
# "bar_box" --> includes countplot for train data, countplot for test data and boxplot for target to each category. it helps us to understand distribution of categories and distribution of target over categories

# "plot_scatter" --> includes scatter plot for target and feature. it shows the correlation coefficient between them and coloring for correlation's degree. it help us to understand relationship between continuous numerical features and target.

# "feature_distribution" --> includes kdeplot, boxplot and probplot for continuous numerical features.

# Defining these functions helps us because machine learning is an iterative process. You need to try different things over and over.


### plotting:

# bar chart
def bar_box(df, col, target = "SalePrice"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex = True)
    
    order = sorted(df[col].unique())
    
    sns.countplot(data = df[df[target].notnull()], x = col, ax = axes[0], order = order)    
    sns.countplot(data = df[df[target].isnull()], x = col, ax = axes[1], order = order)    
    sns.boxplot(data = df, x = col, ax = axes[2], y = target, order = order)
    
    fig.suptitle("For Feature:  " + col)
    axes[0].set_title("in Training Set ")
    axes[1].set_title("in Test Set ")
    axes[2].set_title(col + " --- " + target)
    
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)

for col in cat_cols:
    bar_box(dff, col)



# scatter
def plot_scatter(df, col, target = "SalePrice"):
    import seaborn as sns
    
    sns.set_style("darkgrid")
    
    corr = df[[col, target]].corr()[col][1]    
    c = ["red"] if corr >= 0.7 else (["brown"] if corr >= 0.3 else\
                                    (["lightcoral"] if corr >= 0 else\
                                    (["blue"] if corr <= -0.7 else\
                                    (["royalblue"] if corr <= -0.3 else ["lightskyblue"]))))    

    fig, ax = plt.subplots(figsize = (5, 5))
    
    sns.scatterplot(x = col, y = target, data = df, c = c, ax = ax)        
    ax.set_title("Correlation between " + col + " and " + target + " is: " + str(corr.round(4)))

for col in num_cols:
    plot_scatter(dff, col)


####
def feature_distribution(df, col, target = "SalePrice", test = True):
    sns.set_style("darkgrid")
    if test == True:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        sns.kdeplot(data = df[df[target].notnull()], x = col, fill=True, label = "Train", ax = axes[0], color = "orangered")
        sns.kdeplot(data = df[df[target].isnull()], x = col, fill=True, label = "Test", ax = axes[0], color = "royalblue")
        axes[0].set_title("Distribution")
        axes[0].legend(loc = "best")
        
        sns.boxplot(data = df[df[target].notnull()], y = col, ax = axes[1], color = "orangered")
        sns.boxplot(data = df[df[target].isnull()], y = col, ax = axes[2], color = "royalblue")
        axes[2].set_ylim(axes[1].get_ylim())        
        axes[1].set_title("Boxplot For Train Data")
        axes[2].set_title("Boxplot For Test Data")
        

        stats.probplot(df[df[target].notnull()][col], plot = axes[3])
        stats.probplot(df[df[target].isnull()][col], plot = axes[4])
        axes[4].set_ylim(axes[3].get_ylim())        
        axes[3].set_title("Probability Plot For Train data")
        axes[4].set_title("Probability Plot For Test data")
        
        fig.suptitle("For Feature:  " + col)
    else:
        fig, axes = plt.subplots(1, 3, figsize = (18, 6))
        
        sns.kdeplot(data = df, x = col, fill = True, ax = axes[0], color = "orangered")
        sns.boxplot(data = df, y = col, ax = axes[1], color = "orangered")
        stats.probplot(df[col], plot = axes[2])
        
        axes[0].set_title("Distribution")
        axes[1].set_title("Boxplot")
        axes[2].set_title("Probability Plot")
        fig.suptitle("For Feature:  " + col)

###########
target_encoding = ["MSSubClass", "Neighborhood", "Exterior1st", "Exterior2nd", "Condition1", "Condition2", "HouseStyle"]

for col in target_encoding:
    feature_name = col + "Rank"
    dff3.loc[:, feature_name] = dff3[col].map(dff3.groupby(col).SalePrice.median())
    dff3.loc[:, feature_name] = dff3.loc[:, feature_name].rank(method = "dense")



###### model stack
lgb_model = lgb.LGBMRegressor(colsample_bytree=0.25, learning_rate=0.01,
                              max_depth=13, min_child_samples=7, n_estimators=10000,
                              num_leaves=20, objective='regression', random_state=42,
                              subsample=0.9330025956033094, subsample_freq=1)

xgb_model = xgb.XGBRegressor(colsample_bytree=0.25, gamma=0.0, learning_rate=0.01, max_depth=3,
                             n_estimators=15000, n_jobs=-1, random_state=42, 
                             reg_alpha=0.24206673672530965, reg_lambda=0.40464485640717085, subsample=1.0)

gbr_model = GradientBoostingRegressor(alpha=0.8979588317644014,
                                      learning_rate=0.01, loss='huber',
                                      max_depth=13, max_features=0.1, min_samples_split=109,
                                      n_estimators=10000, n_iter_no_change=100, random_state=42)

svr_model = SVR(C=0.7682824405204463, coef0=0.0001, degree=2, epsilon=0.0001, gamma=0.0042151786393578635, max_iter=10000)

lasso_model = Lasso(alpha=0.00012609086150256233, max_iter=5000, random_state=42)

ridge_model = Ridge(alpha=2.651347536470113, max_iter=5000, random_state=42)

enet_model = ElasticNet(alpha=0.0002286518512853544, l1_ratio=0.6510386358323069, max_iter=5000, random_state=42)


%%time
models = {
    "LGBMRegressor": lgb_model,
    "XGBRegressor": xgb_model,
    "GradientBoostingRegressor": gbr_model,
    "SVR": svr_model,
    "Lasso": lasso_model,
    "Ridge": ridge_model,
#     "ElasticNet": enet_model,
         }

oof_df = pd.DataFrame()
predictions_df = pd.DataFrame()


for name, model in models.items():
    
    print("For model ", name, "\n")
    i = 1
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_test))
    
    for train_ix, test_ix in kf.split(X_train.values):
        
        print("Out of fold predictions generating for fold ", i)
        
        train_X, train_y = X_train.values[train_ix], y_train[train_ix]
        test_X, test_y = X_train.values[test_ix], y_train[test_ix]
        
        if name == "LGBMRegressor":
            model.fit(train_X, train_y,
                      eval_set = [(test_X, test_y)],
                      eval_metric = "rmse",
                      early_stopping_rounds=200,
                      verbose=0)
            
        elif name == "XGBRegressor":
            model.fit(train_X, train_y,
                      eval_set = [(test_X, test_y)],
                      eval_metric = "rmse",
                      early_stopping_rounds=250,
                      verbose=0)
        else:
            model.fit(train_X, train_y)
            
        oof[test_ix] = oof[test_ix] + model.predict(X_train.values[test_ix])
        predictions = predictions + model.predict(X_test.values)
        
        i = i + 1
        
        oof_df[name] = oof
        predictions_df[name] = predictions / 10
        
        
    print("\nDone \n")




oof = np.zeros(len(X_train))
predictions = np.zeros(len(X_test))
i = 1

for train_ix, test_ix in kf.split(oof_df):

    print("Out of fold predictions generating for fold ", i)

    train_X, train_y = oof_df.values[train_ix], y_train[train_ix]
    test_X, test_y = oof_df.values[test_ix], y_train[test_ix]
    
    model = gbr_model
    model.fit(train_X, train_y)

#     model.fit(train_X, train_y,
#                   eval_set = [(test_X, test_y)],
#                   eval_metric = "rmse",
#                   early_stopping_rounds=250,
#                   verbose=0)        

    oof[test_ix] = oof[test_ix] + model.predict(oof_df.values[test_ix])
    predictions = predictions + model.predict(predictions_df)
    
    i = i + 1

    oof_stacked = oof
    stack_preds = predictions / 10      
    
    
    
preds = (4 * stack_preds +
     predictions_df["LGBMRegressor"] +
     predictions_df["XGBRegressor"] +
     2 * predictions_df["GradientBoostingRegressor"] +
     predictions_df["SVR"] +
     predictions_df["Lasso"]) / 10

sub = pd.DataFrame({"Id": test_id.Id, "SalePrice": np.expm1(preds)})
sub.to_csv("BlendedModel120121.csv", index = False)

sub


### stacking:

    
from numba import jit

@jit
def gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini    

def gini_normalized(a, p):
     return gini(a, p) / gini(a, a)
    
def gini_lgb(preds, dtrain):
    y = dtrain
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True    
    
def gini_lgb_train(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True     

    
# Training
n_splits = 3
splitter = model_selection.StratifiedShuffleSplit(n_splits=n_splits)
scores = []

submission = pd.DataFrame({
    'id': test_ids,
    'target': 0
})

params1 = {'learning_rate': 0.09, 'max_depth': 6, 'boosting': 'gbdt', 'objective': 'binary', 
          'metric': 'auc', 'num_leaves': 40, 'min_data_in_leaf': 200, 'max_bin': 100, 'colsample_bytree' : 0.5,   
          'subsample': 0.7, 'subsample_freq': 2, 'verbose':-1, 'is_training_metric': False, 'seed': 1974}
params2 = {'learning_rate': 0.12, 'max_depth': 4, 'verbose':-1, 'num_leaves':16,
           'is_training_metric': False, 'seed': 1974} 
params3 = {'learning_rate': 0.11, 'subsample': 0.8, 'boosting': 'gbdt', 'objective': 'binary', 
          'metric': 'auc', 'subsample_freq': 10, 'colsample_bytree': 0.6, 'max_bin': 10, 
           'min_child_samples': 500,'verbose':-1, 'is_training_metric': False, 'seed': 1974}  

num_models = 3
log_model       = LogisticRegression()
X_logreg_train  = np.zeros((X_train.shape[0], n_splits * num_models))
X_logreg_test   = np.zeros((X_test.shape[0], n_splits * num_models))

lgb_params1 = {}
lgb_params1['learning_rate'] = 0.02
lgb_params1['n_estimators'] = 300
lgb_params1['max_bin'] = 10
lgb_params1['subsample'] = 0.7
lgb_params1['subsample_freq'] = 12
lgb_params1['colsample_bytree'] = 0.7   
lgb_params1['min_child_samples'] = 600
lgb_params1['seed'] = 1974


lgb_params2 = {}
lgb_params2['n_estimators'] = 1500
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 1974


lgb_params3 = {}
lgb_params3['n_estimators'] = 1500
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 1974

model1 = lgb.LGBMClassifier(**lgb_params1)
model2 = lgb.LGBMClassifier(**lgb_params2)
model3 = lgb.LGBMClassifier(**lgb_params3)

for i, (fit_index, val_index) in enumerate(splitter.split(X_train, y_train)):
    X_fit = X_train.iloc[fit_index,:].copy()
    y_fit = y_train.iloc[fit_index].copy()
    X_val = X_train.iloc[val_index,:].copy()
    y_val = y_train.iloc[val_index].copy()

    model1.fit(
        X_fit,
        y_fit)
 #       eval_set=[(X_val, y_val)],
 #       eval_metric=gini_lgb,
 #       early_stopping_rounds=50,
 #       verbose=False  )
 #   model1 = lgb.train(params1, 
 #                 train_set       = lgb.Dataset(X_fit, label=y_fit), 
 #                 num_boost_round = 200,
 #                 valid_sets      = lgb.Dataset(X_val, label=y_val),
 #                 verbose_eval    = 50, 
 #                 feval           = gini_lgb,
 #                 early_stopping_rounds = 50)
    #y_val_predprob1 = model1.predict(X_val, num_iteration=model1.best_iteration)
    y_val_predprob1 = model1.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob1)
    scores.append(score)
    print('Fold {} model 1: {} gini'.format(i+1, score))
    #x_test_pred1   = model1.predict(X_test, num_iteration=model1.best_iteration)
    x_test_pred1   = model1.predict_proba(X_test)[:,1] 
    #x_train_pred1  = model1.predict(X_train, num_iteration=model1.best_iteration)
    x_train_pred1  = model1.predict_proba(X_train)[:,1] 
    X_logreg_test[:, i * num_models]  = x_test_pred1
    X_logreg_train[:, i * num_models] = x_train_pred1
    
    model2.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric=gini_lgb,
        early_stopping_rounds=50,
        verbose=False  )
    y_val_predprob2 = model2.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob2)
    scores.append(score)
    print('Fold {} model 2: {} gini'.format(i+1, score))
    x_test_pred2   = model2.predict_proba(X_test)[:,1] 
    x_train_pred2  = model2.predict_proba(X_train)[:,1] 
    X_logreg_test[:, i * num_models + 1]  = x_test_pred2
    X_logreg_train[:, i * num_models + 1] = x_train_pred2
    
    model3.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric=gini_lgb,
        early_stopping_rounds=50,
        verbose=False  )
    y_val_predprob3 = model3.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob3)
    scores.append(score)
    print('Fold {} model 3: {} gini'.format(i+1, score))
    x_test_pred3   = model3.predict_proba(X_test)[:,1] 
    x_train_pred3  = model3.predict_proba(X_train)[:,1] 
    X_logreg_test[:, i * num_models + 2]  = x_test_pred3
    X_logreg_train[:, i * num_models + 2] = x_train_pred3

log_model.fit(X = X_logreg_train, y = y_train)
submission['target'] = log_model.predict_proba(X_logreg_test)[:,1]

###############

logTransformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

featureTransformer = ColumnTransformer([
        ('log_scaling', logTransformer, ['GrLivArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'LotArea', 'AvgRoomSF', 'Shed', 'TotRmsAbvGrd']),
        ('neighborhood_onehot', OneHotEncoder(categories=[neighborhoodCategories]), ['Neighborhood']),
        ('neighborhood_grp_onehot', OneHotEncoder(), ['NeighborhoodGroups']),
        ('lot_shape_onehot', OneHotEncoder(categories=[lotShapeCategories]), ['LotShape']),
        ('land_slope_onehot', OneHotEncoder(categories=[landSlopeCategories]), ['LandSlope']),
        ('sale_condition_onehot', OneHotEncoder(categories=[saleCondCategories]), ['SaleCondition']),
        ('land_contour_onehot', OneHotEncoder(), ['LandContour']),
        ('zoning_onehot', OneHotEncoder(), ['MSZoning']),
        ('bldg_type_onehot', OneHotEncoder(), ['BldgType']),
        ('masvrn_type_onehot', OneHotEncoder(), ['MasVnrType']),
        ('house_style_onehot', OneHotEncoder(), ['HouseStyle']),
        ('season_onehot', OneHotEncoder(), ['Season']),
    ],
    remainder='passthrough'
)


%%time

from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    max_depth=6,
    n_estimators=8000,
    learning_rate=0.01,
    min_child_weight=1.5,
    subsample=0.2,
    gamma=0.01,
    reg_alpha=1,
    reg_lambda=0.325,
    objective='reg:gamma',
    booster='gbtree'
)

xgb_pipeline = Pipeline([
    ('preprocessing', featureTransformer),
    ('xgb_regressor', xgb_model),
])

print('XGB Regressor:')
score_model(xgb_pipeline, X, Y)




# sklearn's pipeline API is limited at this point and doesn't provide a way to get columns of transformed X array
# This snippet will cover our back 

def get_columns_from_transformer(column_transformer, input_colums):    
    col_name = []

    for transformer_in_columns in column_transformer.transformers_[:-1]: #the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1],Pipeline): 
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names(raw_col_name)
        except AttributeError: # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names,np.ndarray): # eg.
            col_name += names.tolist()
        elif isinstance(names,list):
            col_name += names    
        elif isinstance(names,str):
            col_name.append(names)

    [_, _, reminder_columns] = column_transformer.transformers_[-1]

    for col_idx in reminder_columns:
        col_name.append(input_colums[col_idx])

    return col_name


xgb_pipeline.fit(X, Y)
X_columns = get_columns_from_transformer(xgb_pipeline.named_steps['preprocessing'], list(X.columns))

features_list = sorted(zip(xgb_pipeline.named_steps['xgb_regressor'].feature_importances_, X_columns), reverse=True)
features_list



from eli5.sklearn import PermutationImportance

transformed_X = xgb_pipeline.named_steps['preprocessing'].transform(X)

permutation_importance = PermutationImportance(
    xgb_model, 
    scoring=make_scorer(neg_rmsle),
    cv=2,
    random_state=42,
).fit(transformed_X, Y)

eli5.show_weights(permutation_importance, feature_names=X_columns, top=125)



###############
cv = KFold(n_splits=4, random_state=random_state)

svr = SVR(**svr_params)
ridge = Ridge(**ridge_params, random_state=random_state)
lasso = Lasso(**lasso_params, random_state=random_state)
lgbm = LGBMRegressor(**lgbm_params, random_state=random_state)
rf = RandomForestRegressor(**rf_params, random_state=random_state)
stack = StackingCVRegressor(
    regressors=[svr, ridge, lasso, lgbm, rf],
    meta_regressor=LinearRegression(n_jobs=-1),
    random_state=random_state,
    cv=cv,
    n_jobs=-1,
)

svr_scores = cross_val_score(
    svr, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"
)
ridge_scores = cross_val_score(
    ridge, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"
)
lasso_scores = cross_val_score(
    lasso, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"
)
lgbm_scores = cross_val_score(
    lgbm, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"
)
rf_scores = cross_val_score(
    rf, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"
)
stack_scores = cross_val_score(
    stack, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"
)

scores = [svr_scores, ridge_scores, lasso_scores, lgbm_scores, rf_scores, stack_scores]
models = ["SVR", "RIDGE", "LASSO", "LGBM", "RF", "STACK"]
score_medians = [
    round(np.median([mean for mean in modelscore]), 5) for modelscore in scores
]


