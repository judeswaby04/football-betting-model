import numpy as np
import math
import pandas as pd
from scipy.optimize import minimize

def correlation_function(home_goals, away_goals, lambda_, mu, rho = 0.01):

    # Only 5 cases so we can do it manually as opposed to solving LOTP equations numerically
    if home_goals == 0 and away_goals == 0:
        multiplyer = 1 - lambda_ * mu * rho
        
    elif home_goals == 0 and away_goals == 1:
        multiplyer = 1 + lambda_ * rho

    elif home_goals == 1 and away_goals == 0:
        multiplyer = 1 + mu * rho

    elif home_goals == 1 and away_goals == 1:
        multiplyer = 1 - rho

    else:
        multiplyer = 1
        
    return multiplyer

# must optimise over attack, defence params and the home team advantage gamma
def log_likelihood(params, dataframe, home_goals_and_team, away_goals_and_team, teams, decays, rho = 0.01):
    log_L = 0
    gamma = params[-1]
    for m in range(dataframe.shape[0]):
        home_atk = params[home_goals_and_team[m][1]]
        home_def = params[home_goals_and_team[m][1] + len(teams)]

        away_atk = params[away_goals_and_team[m][1]]
        away_def = params[away_goals_and_team[m][1] + len(teams)]

        lambda_ = gamma * home_atk * away_def
        mu = home_def * away_atk

        home_goals = home_goals_and_team[m][0]
        away_goals = away_goals_and_team[m][0]

        tau = correlation_function(home_goals, away_goals, lambda_, mu, rho)
        if tau <= 0:
            return np.inf
        decay = decays[m]

        log_L += (np.log(tau) + home_goals * np.log(lambda_) - lambda_ + away_goals * np.log(mu) - mu) * decay
    return -log_L


     
def atk_def_params(dataframe, rho, phi = -np.log(0.9)/365.25):
    home_teams = sorted(list(dict.fromkeys(dataframe["home_team"])))
    away_teams = sorted(list(dict.fromkeys(dataframe["away_team"])))
    team_difference = (set(home_teams) - set(away_teams)) | (set(away_teams) - set(home_teams))

    if not team_difference:
        teams = home_teams
    else:
        raise ValueError(f"Dataframe storing incorrect teams, problematic teams are: {team_difference}")
    
    teams_dict = {team : i for i, team in enumerate(teams)}

    home_goals_and_team = []
    away_goals_and_team = []
    decays = []
    # data.py already ordered the dates
    most_recent_match = dataframe["date"].iloc[-1]

    for m in range(dataframe.shape[0]):
        home_goals = dataframe["home_goals"].iloc[m]
        away_goals = dataframe["away_goals"].iloc[m]

        home_idx = teams_dict.get(dataframe["home_team"].iloc[m])
        away_idx = teams_dict.get(dataframe["away_team"].iloc[m])

        # storing index for easier lookup
        home_goals_and_team.append([home_goals, home_idx])
        away_goals_and_team.append([away_goals, away_idx])

        t = (most_recent_match - dataframe["date"].iloc[m]).days
        decays.append(np.exp(-phi * t))

    # length of inital params needs attack params, defence params and a home advantage constant
    initial_params = np.ones(len(teams) * 2 + 1)

    def norm_constraint_function(params):
        atk_coeff_sum = 0
        for i in range(len(teams)):
            atk_coeff_sum += params[i]
        
        return atk_coeff_sum - len(teams)

    optimal_coeffs = minimize(log_likelihood, initial_params, args = (dataframe, home_goals_and_team, away_goals_and_team, teams, decays, rho), bounds = [(1e-6, 2)] * len(initial_params), constraints = {"type": "eq", "fun": norm_constraint_function})

    coeffs = optimal_coeffs.x
    gamma = coeffs[-1]


    att_def_coeffs_dict = {"Team": [team for team in teams], "AttackCoefficient": [coeffs[i] for i in range(len(teams))], "DefenceCoefficient": [coeffs[i+len(teams)] for i in range(len(teams))]}
    df_atk_def_coeffs = pd.DataFrame(att_def_coeffs_dict)

    print(optimal_coeffs.success)
    print(optimal_coeffs.message)



    return df_atk_def_coeffs, float(gamma)


def hda_odds(dataframe, df_atk_def_coeffs, gamma, rho):

    home_odds = []
    draw_odds = []
    away_odds = []

    max_goals = 10

    for match_idx in range(dataframe.shape[0]):

        score_matrix = []

        home_team = dataframe["home_team"].iloc[match_idx]
        away_team = dataframe["away_team"].iloc[match_idx]

        home_atk = df_atk_def_coeffs.loc[df_atk_def_coeffs["Team"] == home_team, "AttackCoefficient"].iloc[0]
        home_def = df_atk_def_coeffs.loc[df_atk_def_coeffs["Team"] == home_team, "DefenceCoefficient"].iloc[0]

        away_atk = df_atk_def_coeffs.loc[df_atk_def_coeffs["Team"] == away_team, "AttackCoefficient"].iloc[0]
        away_def = df_atk_def_coeffs.loc[df_atk_def_coeffs["Team"] == away_team, "DefenceCoefficient"].iloc[0]

        lambda_ = home_atk * away_def * gamma
        mu = home_def * away_atk

        for x in range(max_goals + 1):
            row = []

            for y in range(max_goals + 1):
                tau = correlation_function(x, y, lambda_, mu, rho)
                p = tau * (lambda_ ** x) * (mu ** y) * np.exp(-(lambda_ + mu)) / (math.factorial(x) * math.factorial(y))
                row.append(p)

            score_matrix.append(row)

        score_matrix = np.array(score_matrix)

        home_prob = np.tril(score_matrix, k=-1).sum()
        draw_prob = np.trace(score_matrix)
        away_prob = np.triu(score_matrix, k=1).sum()

        total = home_prob + draw_prob + away_prob

        home_prob = home_prob / total
        draw_prob = draw_prob / total
        away_prob = away_prob / total

        home_odds.append(1 / home_prob)
        draw_odds.append(1 / draw_prob)
        away_odds.append(1 / away_prob)

    df_odds = pd.DataFrame({"Date": dataframe["date"], "HomeTeam": dataframe["home_team"], "AwayTeam": dataframe["away_team"],
                            "Home": home_odds, "Draw": draw_odds, "Away": away_odds})
    
    return df_odds






