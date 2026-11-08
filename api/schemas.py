from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customerID: str = Field(min_length=1, max_length=128)

    gender: Literal["Male", "Female"]

    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0, allow_inf_nan=False)

    TotalCharges: float | None = Field(ge=0, allow_inf_nan=False)

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def normalize_blank_total_charges(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, value, info: ValidationInfo):
        if value is None and info.data.get("tenure") != 0:
            raise ValueError("TotalCharges may be blank only when tenure is 0")
        return value


class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float

class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerInput] = Field(min_length=1, max_length=20)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]