"""
Logloss
Brier
Tune DC
MNLG  c tunng
XGBoost tune (depth, eta, )
"""
import numpy as np
import pandas as pd


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
