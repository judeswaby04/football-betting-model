from xgboost import XGBClassifier
import pandas as pd


def fit_xgboost(train_features, train_results, val_features, val_results, max_depth=3, learning_rate=0.05, reg_lambda=1.0):
    
    result_map = {"A": 0, "D": 1, "H": 2}

    encoded_train = train_results.map(result_map)
    encoded_val = val_results.map(result_map)

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
        train_features,
        encoded_train,
        eval_set=[(val_features, encoded_val)],
        verbose=False
    )

    return model


def predict_probabilities(model, features):
    probabilities = model.predict_proba(features)

    return pd.DataFrame(
        probabilities,
        columns=["A", "D", "H"],
        index=features.index
    )

