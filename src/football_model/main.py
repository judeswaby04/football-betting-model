import numpy as np
import pandas as pd

import data_ as dt
import dix_col as dc
import evaluation as ev
import multinomial as mn
import xgboost_model as xgb
import backtest as bt


paths = [
    "data/prem_8.csv",
    "data/prem_9.csv",
    "data/prem_10.csv",
    "data/prem_11.csv",
    "data/prem_12.csv",
    "data/prem_13.csv",
    "data/prem_14.csv",
    "data/prem_15.csv",
    "data/prem_16.csv",
    "data/prem_17.csv",
    "data/prem_18.csv",
    "data/prem_19.csv",
    "data/prem_20.csv",
    "data/prem_21.csv",
    "data/prem_22.csv",
    "data/prem_23.csv",
    "data/prem_24.csv",
    "data/prem_25.csv",
    "data/prem_26.csv"
]



# split data
df_train_DC, df_val_DC, df_train_ML, df_val_ML, df_test = dt.full_dateframe_and_split(paths)


# tune Dixon-Coles
best_DC_params, best_DC_ll = ev.decay_corr_optimiser(df_train_DC, df_val_DC)

rho = best_DC_params[0]
decay = best_DC_params[1]
phi = -np.log(1 - decay) / 365.25

print("DC:", best_DC_params, best_DC_ll)


# fit DC for ML training data
df_before_train_ML = pd.concat([df_train_DC, df_val_DC], ignore_index=True)

atk_def, gamma = dc.atk_def_params(df_before_train_ML, rho, phi)

df_train_ML, removed_train = dc.remove_unseen_teams(df_train_ML, atk_def)

train_DC_odds = dc.hda_odds(df_train_ML, atk_def, gamma, rho)


# refit DC for ML validation data
df_before_val_ML = pd.concat(
    [df_train_DC, df_val_DC, df_train_ML],
    ignore_index=True
)

atk_def, gamma = dc.atk_def_params(df_before_val_ML, rho, phi)

df_val_ML, removed_val = dc.remove_unseen_teams(df_val_ML, atk_def)

val_DC_odds = dc.hda_odds(df_val_ML, atk_def, gamma, rho)


# tune multinomial
c_vec = [0.01, 0.1, 1, 10, 100]

best_C, best_mn_ll = ev.multinomial_optimiser(
    train_DC_odds,
    val_DC_odds,
    df_train_ML,
    df_val_ML,
    c_vec
)

print("Multinomial:", best_C, best_mn_ll)


# tune XGBoost
depth_vec = [2, 3, 4]
lr_vec = [0.01, 0.05, 0.1]
lambda_vec = [0.1, 1, 10]

best_xgb_params, best_xgb_ll = ev.xgb_optimiser(
    train_DC_odds,
    df_train_ML,
    val_DC_odds,
    df_val_ML,
    depth_vec,
    lr_vec,
    lambda_vec
)

print("XGBoost:", best_xgb_params, best_xgb_ll)


# fit ML models
mn_model = mn.fit_multinomial(
    train_DC_odds,
    df_train_ML,
    best_C
)

xgb_model = xgb.fit_xgboost(
    train_DC_odds,
    df_train_ML,
    val_DC_odds,
    df_val_ML,
    best_xgb_params[0],
    best_xgb_params[1],
    best_xgb_params[2]
)


# fit DC using everything before test
df_before_test = pd.concat(
    [df_train_DC, df_val_DC, df_train_ML, df_val_ML],
    ignore_index=True
)

atk_def, gamma = dc.atk_def_params(df_before_test, rho, phi)

df_test, removed_test = dc.remove_unseen_teams(df_test, atk_def)

test_DC_odds = dc.hda_odds(df_test, atk_def, gamma, rho)


# ML predictions
mn_test_odds = mn.predict_probabilities(
    mn_model,
    test_DC_odds,
    df_test
)

xgb_test_odds = xgb.predict_probabilities(
    xgb_model,
    test_DC_odds,
    df_test
)


# evaluate
print("DC test log loss:", ev.log_loss(df_test, test_DC_odds))
print("MN test log loss:", ev.log_loss(df_test, mn_test_odds))
print("XGB test log loss:", ev.log_loss(df_test, xgb_test_odds))

print("DC test Brier:", ev.brier_score(df_test, test_DC_odds))
print("MN test Brier:", ev.brier_score(df_test, mn_test_odds))
print("XGB test Brier:", ev.brier_score(df_test, xgb_test_odds))


# backtest
print("MN backtest:", bt.backtest(mn_test_odds, df_test))
print("XGB backtest:", bt.backtest(xgb_test_odds, df_test))