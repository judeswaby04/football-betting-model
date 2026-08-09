import pandas as pd

def model_probabilities(df_DC_odds):
    df_DC_probs = pd.DataFrame({})
    adjustment = 1/df_DC_odds["Home"] + 1/df_DC_odds["Draw"] + 1/df_DC_odds["Away"]

    df_DC_probs["model_home"] = 1 / (df_DC_odds["Home"] * adjustment)
    df_DC_probs["model_draw"] = 1 / (df_DC_odds["Draw"] * adjustment)
    df_DC_probs["model_away"] = 1 / (df_DC_odds["Away"] * adjustment)

    return df_DC_probs

def raw_feature_extract(dataframe):
    df_market_probs = pd.DataFrame({})
    adjustment = (1 / dataframe['home_odds']) + (1 / dataframe['draw_odds']) + (1 / dataframe['away_odds'])

    df_market_probs["market_home_probs"] = 1 / (dataframe["home_odds"] * adjustment)
    df_market_probs["market_draw_probs"] = 1 / (dataframe["draw_odds"] * adjustment)
    df_market_probs["market_away_probs"] = 1 / (dataframe["away_odds"] * adjustment)

    df_market_gaps = pd.DataFrame(index=dataframe.index)

    df_market_gaps["home_market_gap"] = 1/dataframe["avg_home_odds"] - 1/dataframe["max_home_odds"]
    df_market_gaps["draw_market_gap"] = 1/dataframe["avg_draw_odds"] - 1/dataframe["max_draw_odds"]
    df_market_gaps["away_market_gap"] = 1/dataframe["avg_away_odds"] - 1/dataframe["max_away_odds"]
    
    dataframe = pd.concat([df_market_probs, df_market_gaps], axis = 1)

    return dataframe

def features(df_DC_odds, dataframe):
    df_match = dataframe[["date", "home_team", "away_team"]]
    df_raw_features = raw_feature_extract(dataframe)
    df_DC_model = model_probabilities(df_DC_odds)

    df_features = pd.concat([df_match, df_raw_features, df_DC_model, dataframe["result"]], axis = 1)

    return df_features

