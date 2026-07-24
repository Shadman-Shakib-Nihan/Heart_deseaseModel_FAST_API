from pydantic import BaseModel, Field, conint, confloat


class HeartInput(BaseModel):
    """
    Input schema for a single patient's data.
    Field names and order match the feature columns the
    model was trained on (all columns in heart.csv except 'target').
    """
    age: conint(ge=1, le=120) = Field(
        ..., example=52, description="Age in years"
    )
    sex: conint(ge=0, le=1) = Field(
        ..., example=1, description="Sex (1 = male, 0 = female)"
    )
    cp: conint(ge=0, le=3) = Field(
        ..., example=0, description="Chest pain type (0-3, 4 possible values)"
    )
    trestbps: conint(ge=0, le=300) = Field(
        ..., example=125, description="Resting blood pressure (mm Hg)"
    )
    chol: conint(ge=0, le=700) = Field(
        ..., example=212, description="Serum cholesterol (mg/dl)"
    )
    fbs: conint(ge=0, le=1) = Field(
        ..., example=0, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)"
    )
    restecg: conint(ge=0, le=2) = Field(
        ..., example=1, description="Resting electrocardiographic results (0, 1, 2)"
    )
    thalach: conint(ge=0, le=250) = Field(
        ..., example=168, description="Maximum heart rate achieved"
    )
    exang: conint(ge=0, le=1) = Field(
        ..., example=0, description="Exercise induced angina (1 = yes, 0 = no)"
    )
    oldpeak: confloat(ge=0.0, le=10.0) = Field(
        ..., example=1.0, description="ST depression induced by exercise relative to rest"
    )
    slope: conint(ge=0, le=2) = Field(
        ..., example=2, description="Slope of the peak exercise ST segment (0-2)"
    )
    ca: conint(ge=0, le=3) = Field(
        ..., example=2, description="Number of major vessels (0-3) colored by flourosopy"
    )
    thal: conint(ge=0, le=3) = Field(
        ..., example=2,
        description="Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 0,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 2,
                "thal": 2
            }
        }


class PredictionOutput(BaseModel):
    """Response schema returned by /predict"""
    heart_disease: bool = Field(..., description="True if the model predicts heart disease, else False")
    probability: float = Field(..., description="Model's predicted probability of heart disease (0-1)")


class ModelInfo(BaseModel):
    """Response schema returned by /info"""
    model_type: str
    features: list[str]