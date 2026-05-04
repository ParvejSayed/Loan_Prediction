import sys
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.exception import CustomException
from src.logger import logging
from src.pipeline.prediction_pipeline import customClass, predictionPipeline

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts whether a bank loan application will be approved using a Gradient Boosting model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoanApplication(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: int = Field(..., gt=0, description="Monthly income of the applicant")
    CoapplicantIncome: float = Field(..., ge=0, description="Monthly income of the co-applicant")
    LoanAmount: float = Field(..., gt=0, description="Loan amount requested (in thousands)")
    Loan_Amount_Term: float = Field(..., gt=0, description="Loan repayment term in months")
    Credit_History: float = Field(..., ge=0, le=1, description="1 = good credit history, 0 = bad")
    Property_Area: Literal["Urban", "Semiurban", "Rural"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "1",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 1500.0,
                "LoanAmount": 120.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0,
                "Property_Area": "Urban",
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    status: str
    message: str


@app.get("/", tags=["Health"])
def root():
    return {"message": "Loan Approval Prediction API is running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: LoanApplication):
    try:
        logging.info(f"Received prediction request: {application.model_dump()}")

        data = customClass(**application.model_dump())
        df = data.get_data_frame()

        pipeline = predictionPipeline()
        pred = pipeline.predict(df)
        result = int(pred[0])

        if result == 1:
            status = "Approved"
            message = "Congratulations! Your Loan is APPROVED."
        else:
            status = "Rejected"
            message = "Sorry, Your Loan is NOT APPROVED. Better luck next time!"

        logging.info(f"Prediction: {status}")
        return PredictionResponse(prediction=result, status=status, message=message)

    except CustomException as e:
        logging.error(f"CustomException: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
