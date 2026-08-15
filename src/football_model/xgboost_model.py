from xgboost import XGBClassifier
import pandas as pd
import features as ft


FEATURE_COLS = [
    "market_home_probs",
    "market_draw_probs",
    "market_away_probs",
    "home_market_gap",
    "draw_market_gap",
    "away_market_gap",
    "model_home",
    "model_draw",
    "model_away"
]


def fit_xgboost(train_DC_odds, train_dataframe, val_DC_odds, val_dataframe, max_depth=3, learning_rate=0.05, reg_lambda=1.0):
    
    train_features = ft.features(train_DC_odds, train_dataframe)
    val_features = ft.features(val_DC_odds, val_dataframe)

    X_train = train_features[FEATURE_COLS]
    X_val = val_features[FEATURE_COLS]

    result_map = {"A": 0, "D": 1, "H": 2}

    y_train = train_features["result"].map(result_map)
    y_val = val_features["result"].map(result_map)

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        n_estimators=1000,
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=50,
        random_state=1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def predict_probabilities(model, df_DC_odds, dataframe):
    df_features = ft.features(df_DC_odds, dataframe)

    X = df_features[FEATURE_COLS]

    probabilities = model.predict_proba(X)

    df_probabilities = pd.DataFrame(
        probabilities,
        columns=["A", "D", "H"],
        index=dataframe.index
    )

    df_odds = pd.DataFrame({
        "Home": 1 / df_probabilities["H"],
        "Draw": 1 / df_probabilities["D"],
        "Away": 1 / df_probabilities["A"]
    })

    return df_odds

