"""
Logloss
Brier
Tune DC
MNLG  c tunng
XGBoost tune (depth, eta, )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dix_col as dc
import features as ft
import multinomial as mn
import xgboost_model as xgb


def log_loss(dataframe, df_odds):
    logloss = 0
    home_probs = 1/df_odds["Home"]
    draw_probs = 1/df_odds["Draw"]
    away_probs = 1/df_odds["Away"]

    results = dataframe["result"]

    n_matches = results.shape[0]

    for i in range(n_matches):
        if results.iloc[i] == "H":
            logloss -= np.log(home_probs.iloc[i])

        elif results.iloc[i] == "D":
            logloss -= np.log(draw_probs.iloc[i])
        
        elif results.iloc[i] == "A":
            logloss -= np.log(away_probs.iloc[i])

        else:
            raise ValueError(f"{dataframe["result"].iloc[i]} is not a valid result")
        
    
    return logloss/n_matches


def brier_score(dataframe, df_odds):
    brier_score = 0
    home_probs = 1/df_odds["Home"]
    draw_probs = 1/df_odds["Draw"]
    away_probs = 1/df_odds["Away"]

    results = dataframe["result"]

    n_matches = results.shape[0]

    for i in range(n_matches):
        true_vec = np.array([results.iloc[i] == "H", results.iloc[i] == "D", results.iloc[i] == "A"])
        model_vec = np.array([home_probs.iloc[i], draw_probs.iloc[i], away_probs.iloc[i]])
        brier_score += np.linalg.norm(true_vec - model_vec) ** 2

    return brier_score/n_matches


def calibration_table(dataframe, df_odds):
    bins = np.linspace(0.0, 1.0, 11)

    home_probs = 1 / df_odds["Home"]
    draw_probs = 1 / df_odds["Draw"]
    away_probs = 1 / df_odds["Away"]

    binsH = pd.cut(home_probs, bins, include_lowest=True)
    binsD = pd.cut(draw_probs, bins, include_lowest=True)
    binsA = pd.cut(away_probs, bins, include_lowest=True)

    results = dataframe["result"]

    home_win = (results == "H").astype(int)
    draw = (results == "D").astype(int)
    away_win = (results == "A").astype(int)

    home_table = pd.DataFrame({"bin": binsH, "actual": home_win})
    draw_table = pd.DataFrame({"bin": binsD, "actual": draw})
    away_table = pd.DataFrame({"bin": binsA, "actual": away_win})

    result_tracker = pd.DataFrame({
        "H": home_table.groupby("bin", observed=False)["actual"].mean(),
        "D": draw_table.groupby("bin", observed=False)["actual"].mean(),
        "A": away_table.groupby("bin", observed=False)["actual"].mean()
    })

    return result_tracker

def plot_calibration_curves(dataframe, df_odds):
    x = np.linspace(0.05, 0.95, 10)

    result_tracker = calibration_table(dataframe, df_odds)

    plt.plot(x, result_tracker["H"], label="Home")
    plt.plot(x, result_tracker["D"], label="Draw")
    plt.plot(x, result_tracker["A"], label="Away")
    plt.plot(x, x, color="red", linestyle="--", label="Perfect calibration")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    plt.show()


def decay_corr_optimiser(df_DC_train, df_DC_valid, max_rho = 0.1, min_rho = -0.1, max_decay = 0.999, min_decay = 0, grid_size = 10):

    log_losses = {}

    rhos = np.linspace(min_rho, max_rho, grid_size)
    decays = np.linspace(min_decay, max_decay, grid_size)

    for rho in rhos:
        for decay in decays:

            atk_def_coeffs = dc.atk_def_params(df_DC_train, rho,-np.log(1 - decay)/365.25)
            df_hda_odds = dc.hda_odds(df_DC_valid, atk_def_coeffs[0], atk_def_coeffs[1], rho)

            log_losses[(rho,decay)] = log_loss(df_DC_valid, df_hda_odds)

    best_params = min(log_losses, key = log_losses.get)
    best_loss = log_losses[best_params]

    return best_params, best_loss


def multinomial_optimiser(df_train_DC_odds, df_valid_DC_odds, df_train_ML, df_valid_ML, c_vec):
    ll_dict = {}

    for C in c_vec:
        model = mn.fit_multinomial(df_train_DC_odds, df_train_ML, C)
        df_probs = mn.predict_probabilities(model, df_valid_DC_odds, df_valid_ML)
        logloss = log_loss(df_valid_ML, df_probs)
        ll_dict[C] = logloss
    
    best_C = min(ll_dict, key= ll_dict.get)
    best_ll = ll_dict[best_C]
    return best_C, best_ll


def xgb_optimiser(train_DC_odds, df_train_ML, val_DC_odds, df_val_ML, depth_vec, lr_vec, lambda_vec):
    xgb_ll_dict = {}

    for depth in depth_vec:
        for lr in lr_vec:
            for lambda_ in lambda_vec:

                model = xgb.fit_xgboost(train_DC_odds, df_train_ML, val_DC_odds, df_val_ML, depth, lr, lambda_)

                val_odds = xgb.predict_probabilities(model, val_DC_odds, df_val_ML)

                ll = log_loss(df_val_ML, val_odds)

                xgb_ll_dict[(depth, lr, lambda_)] = ll

    best_params = min(xgb_ll_dict, key=xgb_ll_dict.get)
    best_ll = xgb_ll_dict[best_params]

    return best_params, best_ll


# def hda_odds(dataframe, df_atk_def_coeffs, gamma, rho=0.01)