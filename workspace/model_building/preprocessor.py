import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95, random_state=42):
        self.variance_threshold = variance_threshold
        self.random_state = random_state

        self.scaler = PowerTransformer(method='yeo-johnson')
        self.pca = PCA(n_components=variance_threshold)
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination='auto',
            random_state=random_state
        )

    def fit(self, X, y=None):
        # X is already feature-only dataframe
        self.features = X.columns.tolist()

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        X_pca = self.pca.fit_transform(X_scaled)

        # Isolation Forest
        self.iso_forest.fit(X_pca)

        return self

    def transform(self, X):
        # Ensure same column order
        X = X[self.features]

        # Scale
        X_scaled = self.scaler.transform(X)

        # PCA
        X_pca = self.pca.transform(X_scaled)

        X_pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
        )

        # Isolation Forest
        anomalies = self.iso_forest.predict(X_pca)
        X_pca_df['Anomaly'] = anomalies

        return X_pca_df
