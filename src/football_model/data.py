from pathlib import Path
import pandas as pd
from datetime import datetime

# previous used pinnacle, but earlier seasons don't have
REQUIRED_COLS = ["date", "home_team", "away_team", "home_goals", "away_goals", "result", "home_odds", 
                "draw_odds", "away_odds", "max_home_odds", "max_draw_odds", "max_away_odds",
                "avg_home_odds", "avg_draw_odds", "avg_away_odds"]


COL_RENAME_MAP = {"Date" : "date", "HomeTeam": "home_team", "AwayTeam" : "away_team",
                 "FTHG" : "home_goals", "FTAG" : "away_goals", "FTR" : "result",
                 "B365H" : "home_odds", "B365D" : "draw_odds", "B365A" : "away_odds", 
                 "BbMxH" : "max_home_odds", "BbMxD" : "max_draw_odds", "BbMxA" : "max_away_odds",
                 "BbAvH" : "avg_home_odds", "BbAvD" : "avg_draw_odds", "BbAvA" : "avg_away_odds"}

def check_cols_exist(dataframe):
    missing_cols = [col for col in REQUIRED_COLS if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def read_in(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    dataframe = pd.read_csv(path)
    dataframe = dataframe.rename(columns=COL_RENAME_MAP)
    check_cols_exist(dataframe)

    dataframe = dataframe[REQUIRED_COLS]

    return dataframe

# makes dates in to datetime and also soleves the earlier season date format problem
def parse_dates(dataframe):
    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        format="mixed",
        dayfirst=True
    )

    return dataframe


def handle_missing_vals(dataframe):
    rows_with_missing = dataframe.isna().any(axis = 1)
    missing_rows = dataframe[rows_with_missing]
    dataframe = dataframe[~rows_with_missing].copy()

    for i in range(missing_rows.shape[0]):
        row = missing_rows.iloc[i]
        print(f"{row["home_team"]} vs {row["away_team"]} on {row["date"]} \
                    was omitted due to insufficient data")

    return dataframe, missing_rows

def convert_types(dataframe):

    dataframe["home_goals"] = dataframe["home_goals"].astype(int)
    dataframe["away_goals"] = dataframe["away_goals"].astype(int)

    dataframe["home_team"] = dataframe["home_team"].astype(str)
    dataframe["away_team"] = dataframe["away_team"].astype(str)

    dataframe["home_odds"] = dataframe["home_odds"].astype(float)
    dataframe["draw_odds"] = dataframe["draw_odds"].astype(float)
    dataframe["away_odds"] = dataframe["away_odds"].astype(float)

    return dataframe

# Sanity checking everything
def validate_values(dataframe):
    invalid_results = ~dataframe["result"].isin(["H", "D", "A"])
    if invalid_results.any():
        raise ValueError("Incorrectly stored results")
     
    if (dataframe["home_goals"] < 0).any():
        raise ValueError("negative goals")
     
    if (dataframe["away_goals"] < 0).any():
        raise ValueError("negative goals")
    
    invalid_score = (((dataframe["home_goals"] > dataframe["away_goals"]) & (dataframe["result"] != "H"))) | \
        (((dataframe["home_goals"] == dataframe["away_goals"]) & (dataframe["result"] != "D"))) | \
        (((dataframe["home_goals"] < dataframe["away_goals"]) & (dataframe["result"] != "A")))
    
    if invalid_score.any():
        match = dataframe[invalid_score].iloc[0]
        raise ValueError(f"Result doesn't agree with score on match {match["home_team"]} vs {match["away_team"]} on {match["date"]}")
    
    invalid_odds = ((dataframe["home_odds"] < 1) | (dataframe["draw_odds"] < 1) | (dataframe["away_odds"] < 1))

    if invalid_odds.any():
        match = dataframe[invalid_odds].iloc[0]
        raise ValueError(f"Odds on match: {match["home_team"]} vs {match["away_team"]} on {match["date"]} contradict betting")
    
    return "Sanity Checks: Pass"

# helps to be able to call by season
def add_season(dataframe):
    df = dataframe.sort_values("date").copy()
    year = df["date"].iloc[0].year + 1
    season = f"{(year - 1) % 100:02d}/{year % 100:02d}"
    dataframe["season"] = season
    return dataframe

# should serve as final output bringing together all functions and returning train, test and validation
def full_dateframe_and_split(paths, n_val_DC_seasons=1, n_train_ML_seasons=2, n_val_ML_seasons=2, n_test_seasons=1):
    train_DC_seasons = []
    val_DC_seasons = []
    train_ML_seasons = []
    val_ML_seasons = []
    test_seasons = []

    n_train_DC_seasons = len(paths) - n_val_DC_seasons - n_train_ML_seasons - n_val_ML_seasons - n_test_seasons

    if n_train_DC_seasons <= 0:
        raise ValueError("Not enough seasons for requested split")

    for i, path in enumerate(paths):
        dataframe = read_in(path)
        dataframe, missing_vals = handle_missing_vals(dataframe)
        dataframe = convert_types(dataframe)
        dataframe = parse_dates(dataframe)
        dataframe = add_season(dataframe)
        validate_values(dataframe)

        if i < n_train_DC_seasons:
            train_DC_seasons.append(dataframe)

        elif i < n_train_DC_seasons + n_val_DC_seasons:
            val_DC_seasons.append(dataframe)

        elif i < n_train_DC_seasons + n_val_DC_seasons + n_train_ML_seasons:
            train_ML_seasons.append(dataframe)

        elif i < n_train_DC_seasons + n_val_DC_seasons + n_train_ML_seasons + n_val_ML_seasons:
            val_ML_seasons.append(dataframe)

        else:
            test_seasons.append(dataframe)

    df_train_DC = pd.concat(train_DC_seasons, ignore_index=True)
    df_val_DC = pd.concat(val_DC_seasons, ignore_index=True)
    df_train_ML = pd.concat(train_ML_seasons, ignore_index=True)
    df_val_ML = pd.concat(val_ML_seasons, ignore_index=True)
    df_test = pd.concat(test_seasons, ignore_index=True)

    df_train_DC = df_train_DC.sort_values("date").reset_index(drop=True)
    df_val_DC = df_val_DC.sort_values("date").reset_index(drop=True)
    df_train_ML = df_train_ML.sort_values("date").reset_index(drop=True)
    df_val_ML = df_val_ML.sort_values("date").reset_index(drop=True)
    df_test = df_test.sort_values("date").reset_index(drop=True)

    return df_train_DC, df_val_DC, df_train_ML, df_val_ML, df_test



       

     


      
