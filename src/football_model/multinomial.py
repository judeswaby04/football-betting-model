from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import features as ft


FEATURE_COLS = ["market_home_probs", "market_draw_probs", "market_away_probs", 
                "home_market_gap", "draw_market_gap", "away_market_gap",
                "model_home", "model_draw", "model_away"]


def fit_multinomial(df_DC_odds, dataframe, C=1.0):
    df_features = ft.features(df_DC_odds, dataframe)

    X = df_features[FEATURE_COLS]
    y = df_features["result"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=1000
        ))
    ])

    model.fit(X, y)

    return model


def predict_probabilities(model, df_DC_odds, dataframe):
    df_features = ft.features(df_DC_odds, dataframe)

    X = df_features[FEATURE_COLS]

    probabilities = model.predict_proba(X)

    classes = model.named_steps["model"].classes_

    df_probabilities = pd.DataFrame(
        probabilities,
        columns=classes,
        index=dataframe.index
    )

    df_odds = pd.DataFrame({
            "Home": 1 / df_probabilities["H"],
            "Draw": 1 / df_probabilities["D"],
            "Away": 1 / df_probabilities["A"]
        })
    
    return df_odds