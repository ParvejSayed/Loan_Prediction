import os
import sys
import pickle
import pandas as pd

from src.logger import logging
from src.exception import CustomException

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")

_DEP_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}
_AREA_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


class customClass:
    def __init__(
        self,
        Gender: str,
        Married: str,
        Dependents: str,
        Education: str,
        Self_Employed: str,
        ApplicantIncome: int,
        CoapplicantIncome: float,
        LoanAmount: float,
        Loan_Amount_Term: float,
        Credit_History: float,
        Property_Area: str,
    ):
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area

    def get_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Gender": self.Gender,
            "Married": self.Married,
            "Dependents": self.Dependents,
            "Education": self.Education,
            "Self_Employed": self.Self_Employed,
            "ApplicantIncome": self.ApplicantIncome,
            "CoapplicantIncome": self.CoapplicantIncome,
            "LoanAmount": self.LoanAmount,
            "Loan_Amount_Term": self.Loan_Amount_Term,
            "Credit_History": self.Credit_History,
            "Property_Area": self.Property_Area,
        }])


class predictionPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join(ARTIFACTS_DIR, "final_preprocessor.pkl")
        self.model_path = os.path.join(ARTIFACTS_DIR, "gradient_boosting_best_model.pkl")
        self.kmeans_path = os.path.join(ARTIFACTS_DIR, "kmeans_model.pkl")
        self.risk_map_path = os.path.join(ARTIFACTS_DIR, "cluster_risk_map.pkl")

    def _add_cluster_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns cluster and appends risk_probability if KMeans artifacts exist."""
        if not (os.path.exists(self.kmeans_path) and os.path.exists(self.risk_map_path)):
            logging.info("KMeans artifacts not found — skipping cluster feature engineering")
            return df

        kmeans = _load_pickle(self.kmeans_path)
        risk_map = _load_pickle(self.risk_map_path)

        cluster_input = pd.DataFrame({
            "ApplicantIncome": df["ApplicantIncome"],
            "CoapplicantIncome": df["CoapplicantIncome"],
            "Credit_History": df["Credit_History"].fillna(df["Credit_History"].mean()),
            "Dependents_enc": df["Dependents"].map(_DEP_MAP).fillna(0),
            "Property_Area_enc": df["Property_Area"].map(_AREA_MAP).fillna(0),
        })

        cluster_id = int(kmeans.predict(cluster_input)[0])
        df = df.copy()
        df["risk_probability"] = risk_map.get(cluster_id, 0.5)
        logging.info(f"Cluster assigned: {cluster_id}, risk_probability: {df['risk_probability'].iloc[0]}")
        return df

    def predict(self, df: pd.DataFrame):
        try:
            logging.info("Prediction pipeline started")
            df = self._add_cluster_features(df)

            preprocessor = _load_pickle(self.preprocessor_path)
            model = _load_pickle(self.model_path)

            transformed = preprocessor.transform(df)
            prediction = model.predict(transformed)
            logging.info(f"Prediction result: {prediction}")
            return prediction

        except Exception as e:
            raise CustomException(e, sys)
