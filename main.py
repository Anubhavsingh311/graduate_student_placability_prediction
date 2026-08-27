from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Graduate Student Placability Prediction API",
    description="API for predicting graduate student placement",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    print("Logistic Regression model loaded successfully.")
    print("Feature columns loaded successfully.")

except Exception as e:

    print("Error loading model files:")
    print(e)

    model = None
    feature_columns = None


# ============================================================
# INPUT MODEL
# ============================================================

class PlacementInput(BaseModel):

    gender: str

    tenth_percent: float
    tenth_board: str

    twelfth_percent: float
    twelfth_board: str
    twelfth_stream: str

    degree_percent: float
    degree: str

    experience: str

    employment_test_percent: float

    mba_stream: str
    mba_percent: float


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data: PlacementInput):

    df = pd.DataFrame([{

        "gender": data.gender,

        "10th Percent": data.tenth_percent,
        "10th Board": data.tenth_board,

        "12th Percent": data.twelfth_percent,
        "12th Board": data.twelfth_board,
        "12th Stream": data.twelfth_stream,

        "Degree Percent": data.degree_percent,
        "Degree": data.degree,

        "Experience": data.experience,

        "emp_test_percent": data.employment_test_percent,

        "MBA Stream": data.mba_stream,
        "MBA percent": data.mba_percent

    }])


    # ========================================================
    # SAME FEATURE ENGINEERING AS NOTEBOOK
    # ========================================================

    # 1. Academic Average

    df["Academic Average"] = (
        df["10th Percent"]
        + df["12th Percent"]
        + df["Degree Percent"]
    ) / 3


    # 2. School Average

    df["School Average"] = (
        df["10th Percent"]
        + df["12th Percent"]
    ) / 2


    # 3. Higher Education Average

    df["Higher Education Average"] = (
        df["Degree Percent"]
        + df["MBA percent"]
    ) / 2


    # 4. Overall Score

    df["Overall Score"] = (
        df["10th Percent"]
        + df["12th Percent"]
        + df["Degree Percent"]
        + df["emp_test_percent"]
        + df["MBA percent"]
    ) / 5


    # 5. High Score Count

    df["High Score Count"] = (
        (df["10th Percent"] >= 70).astype(int)
        + (df["12th Percent"] >= 70).astype(int)
        + (df["Degree Percent"] >= 70).astype(int)
        + (df["emp_test_percent"] >= 70).astype(int)
        + (df["MBA percent"] >= 70).astype(int)
    )


    # 6. Experience Binary

    df["Experience Binary"] = df["Experience"].map({
        "Yes": 1,
        "No": 0
    })


    # 7. Degree - 10th

    df["Degree - 10th"] = (
        df["Degree Percent"]
        - df["10th Percent"]
    )


    # 8. Degree - 12th

    df["Degree - 12th"] = (
        df["Degree Percent"]
        - df["12th Percent"]
    )


    return df


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(data: PlacementInput):

    # Create engineered features
    df = create_features(data)


    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    # EXACTLY the same method used in the notebook

    df = pd.get_dummies(
        df,
        drop_first=True
    )


    # --------------------------------------------------------
    # Convert boolean columns to numeric
    # --------------------------------------------------------

    df = df.astype(float)


    # --------------------------------------------------------
    # Match training columns
    # --------------------------------------------------------

    # feature_columns.pkl contains the exact columns
    # used when training the Logistic Regression model.

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )


    return df


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Graduate Student Placability Prediction API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "error",
            "message": "Model could not be loaded"
        }

    return {

        "status": "healthy",

        "model": "Logistic Regression",

        "features": len(feature_columns)

    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: PlacementInput):

    try:

        # ----------------------------------------------------
        # Validate percentages
        # ----------------------------------------------------

        percentage_fields = {

            "10th Percent": data.tenth_percent,

            "12th Percent": data.twelfth_percent,

            "Degree Percent": data.degree_percent,

            "Employment Test Percent":
                data.employment_test_percent,

            "MBA Percent":
                data.mba_percent
        }


        for field, value in percentage_fields.items():

            if value < 0 or value > 100:

                raise HTTPException(
                    status_code=400,
                    detail=f"{field} must be between 0 and 100"
                )


        # ----------------------------------------------------
        # Validate experience
        # ----------------------------------------------------

        if data.experience not in ["Yes", "No"]:

            raise HTTPException(
                status_code=400,
                detail="Experience must be either Yes or No"
            )


        # ----------------------------------------------------
        # PREPROCESS INPUT
        # ----------------------------------------------------

        processed_data = preprocess(data)


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            processed_data
        )[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            processed_data
        )[0]


        not_placed_probability = (
            probabilities[0] * 100
        )

        placed_probability = (
            probabilities[1] * 100
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if int(prediction) == 1:

            result = "Placed"

        else:

            result = "Not Placed"


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "prediction": int(prediction),

            "result": result,

            "probability": {

                "placed": round(
                    placed_probability,
                    2
                ),

                "not_placed": round(
                    not_placed_probability,
                    2
                )
            },

            "model": "Logistic Regression"

        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
