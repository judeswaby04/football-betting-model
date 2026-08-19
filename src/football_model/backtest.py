import pandas as pd


def expected_value(model_odds, df_test):
    df_EV = pd.DataFrame({})

    df_EV["Home_EV"] = (df_test["home_odds"] / model_odds["Home"]) - 1
    df_EV["Draw_EV"] = (df_test["draw_odds"] / model_odds["Draw"]) - 1
    df_EV["Away_EV"] = (df_test["away_odds"] / model_odds["Away"]) - 1

    return df_EV

def select_bets(model_odds, df_test, edge = 0.05):

    df_EV = expected_value(model_odds, df_test)

    df_bets = pd.DataFrame({})

    df_bets["Bet_Home"] = (df_EV["Home_EV"] >= edge).astype(int)
    df_bets["Bet_Draw"] = (df_EV["Draw_EV"] >= edge).astype(int)
    df_bets["Bet_Away"] = (df_EV["Away_EV"] >= edge).astype(int)

    return df_bets


def settle_bets(df_test, df_bets):

    # I"m assuming a unit stake, will maybe try Kelly / frac Kelly l8r
    home_returns = 0
    draw_returns = 0
    away_returns = 0

    for i in range(df_bets.shape[0]):

        if df_bets["Bet_Home"].iloc[i] == 1:
            home_returns += int(df_test["result"].iloc[i] == "H") * df_test["home_odds"].iloc[i] - 1

        if df_bets["Bet_Draw"].iloc[i] == 1:
            draw_returns += int(df_test["result"].iloc[i] == "D") * df_test["draw_odds"].iloc[i] - 1
        
        if df_bets["Bet_Away"].iloc[i] == 1:
            away_returns += int(df_test["result"].iloc[i] == "A") * df_test["away_odds"].iloc[i] - 1

    return home_returns + draw_returns + away_returns

def performance_metrics(df_test, df_bets):

    n_bets = df_bets["Bet_Home"].sum() + df_bets["Bet_Draw"].sum() + df_bets["Bet_Away"].sum()

    profit = settle_bets(df_test, df_bets)

    roi = profit / n_bets

    n_wins = ((df_bets["Bet_Home"] == 1) & (df_test["result"] == "H")).sum() + \
            ((df_bets["Bet_Draw"] == 1) & (df_test["result"] == "D")).sum() + \
            ((df_bets["Bet_Away"] == 1) & (df_test["result"] == "A")).sum()

    hit_rate = n_wins / n_bets

    return n_bets, profit, roi, hit_rate


def backtest(model_odds, df_test, edge = 0.05):

    df_bets = select_bets(model_odds, df_test, edge)

    return performance_metrics(df_test, df_bets)
