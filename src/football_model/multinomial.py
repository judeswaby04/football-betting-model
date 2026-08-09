from xgboost import XGBClassifier


def fit_xgboost(features, results):
    result_map = {"A": 0, "D": 1, "H": 2}
    encoded_results = results.map(result_map)

    model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=42)

    model.fit(features, encoded_results)
    return model


def predict_probabilities(model, features):
    return model.predict_proba(features)