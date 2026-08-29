import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

class CreditRiskEngine:
    def __init__(self, data_url=None):
        self.data_url = data_url or "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        self.model = LogisticRegression(max_iter=1000, penalty="l2", C=0.1, solver="lbfgs")
        self.features = []

    def load_and_preprocess(self):
        df = pd.read_excel(self.data_url, header=1)
        df = df.rename(columns={"default payment next month": "DEFAULT"})
        self.features = [c for c in df.columns if c not in ["ID", "DEFAULT"]]

        X_train, X_test, y_train, y_test = train_test_split(
            df[self.features], df["DEFAULT"], test_size=0.30, random_state=42, stratify=df["DEFAULT"]
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, X_test, y_train, y_test):
        train_probs = self.model.predict_proba(X_train)[:, 1]
        test_probs = self.model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, test_probs)
        goods = test_probs[y_test == 0]
        bads = test_probs[y_test == 1]
        ks_stat, _ = ks_2samp(goods, bads)
        brier = brier_score_loss(y_test, test_probs)

        train_binned, bins = pd.qcut(train_probs, q=10, retbins=True, duplicates="drop")
        test_binned = pd.cut(test_probs, bins=bins, include_lowest=True)
        train_dist = train_binned.value_counts(normalize=True).sort_index()
        test_dist = test_binned.value_counts(normalize=True).sort_index()
        psi = np.sum((test_dist - train_dist) * np.log((test_dist + 1e-5) / (train_dist + 1e-5)))

        test_df = X_test.copy()
        test_df["prob"] = test_probs
        test_df["approved"] = (test_df["prob"] < 0.25).astype(int)
        sex_approval = test_df.groupby("SEX")["approved"].mean()
        max_rate = sex_approval.max()
        disparate_impact_ratio = sex_approval.min() / max_rate

        return {
            "model_metadata": {
                "model_type": "Logistic Regression (L2)",
                "dataset": "UCI Default of Credit Card Clients",
                "sample_size_test": len(y_test)
            },
            "performance_metrics": {
                "ROC_AUC": round(float(auc), 4),
                "KS_Statistic": round(float(ks_stat), 4),
                "Brier_Score": round(float(brier), 4),
                "PSI": round(float(psi), 4)
            },
            "fair_lending_audit": {
                "protected_attribute": "SEX",
                "approval_rates": {int(k): round(float(v), 4) for k, v in sex_approval.items()},
                "disparate_impact_ratio": round(float(disparate_impact_ratio), 4),
                "four_fifths_violation": bool(disparate_impact_ratio < 0.80)
            }
        }
