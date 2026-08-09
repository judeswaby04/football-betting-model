from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def fit_multinomial(features, results, C=1.0):
    model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=C))])

    model.fit(features, results)
    return model

def predict_probabilities(model, dataframe):
    return model.predict_proba(dataframe)