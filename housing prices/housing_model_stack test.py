# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:22:22 2021

@author: thoma
"""

# https://www.kaggle.com/orhankaramancode/ensemble-stacked-regressors-top-3-92-acc

#Due to high feature/low sample size of our data, there's a great chance of overfitting. 
#Models with regularization mechanism such as Lasso and Ridge do well in regards to such data. 
#In addition Lasso and Ridge, I'll use SVR, LGBM and RandomForest regressor.

# Optuna's search algorithm

#Initially, I also was concerned regarding the existence of multicollinearity. I tried to tackle the issue by introducing PCA, MCA (realizing that there are highly correlated categorical variables). At some point I even calculated VIF for each variable, and tied the results to feature importance tables to identify high VIF variables that are low in importance, and tried to eliminate those. The results were, in most cases, not what I was looking for - lower LB score and not any improvement on accuracy metrics. A number of articles I found online suggested that multicollinearity is more of a problem for ML explain-ability and model simplicity, but less of a problem for the actual model accuracy. (Not sure to what extent this is true)
#Here's an interesting find: https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/


### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py

import optuna
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold


def objective(trial):
    _C = trial.suggest_float("C", 0.1, 0.5)
    _epsilon = trial.suggest_float("epsilon", 0.01, 0.1)
    _coef = trial.suggest_float("coef0", 0.5, 1)

    svr = SVR(cache_size=5000, kernel="poly", C=_C, epsilon=_epsilon, coef0=_coef)

    score = cross_val_score(
        svr, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
    ).mean()
    return score


optuna.logging.set_verbosity(0)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

svr_params = study.best_params
svr_best_score = study.best_value
print(f"Best score:{svr_best_score} \nOptimized parameters: {svr_params}")


### + other

#######

rf_params = {"max_depth": 8, "max_features": 40, "n_estimators": 132}
svr_params = {
    "kernel": "poly",
    "C": 0.053677105521141605,
    "epsilon": 0.03925943476562099,
    "coef0": 0.9486751042886584,
}
ridge_params = {
    "alpha": 0.9999189637151178,
    "tol": 0.8668539399622242,
    "solver": "cholesky",
}
lasso_params = {"alpha": 0.0004342843645993161, "selection": "random"}
lgbm_params = {
    "num_leaves": 16,
    "max_depth": 6,
    "learning_rate": 0.16060612646519587,
    "n_estimators": 64,
    "min_child_weight": 0.4453842422224686,
}

########


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


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