from sklearn.preprocessing import StandardScaler


def select_features(df):
    X = df[["Age", "Annual_Income", "Spending_Score"]]
    return X


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler