# Heart Disease Prediction API

A FastAPI service that loads a trained heart disease prediction model and exposes inference endpoints.

The project uses a saved scikit-learn pipeline (`model/heart_disease_model.joblib`) trained on `heart.csv` and accepts structured patient data to predict whether the patient is likely to have heart disease.

## Project Structure

- `app/main.py` - FastAPI application and endpoint logic
- `app/schemas.py` - Pydantic request/response schemas
- `model/heart_disease_model.joblib` - saved trained model pipeline
- `model/heart.csv` - original dataset used for training
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image build instructions
- `docker-compose.yml` - service definition for app and Redis

## Features

- `GET /health` - health check endpoint
- `GET /info` - returns model type and expected feature list
- `POST /predict` - returns `heart_disease` prediction and probability

## Requirements

- Python 3.11 compatible environment
- `pip` for installing requirements
- `uvicorn` to run the FastAPI server

## Install and Run Locally

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 5000
```

4. Open the interactive docs:

- Swagger UI: `http://127.0.0.1:5000/docs`
- ReDoc: `http://127.0.0.1:5000/redoc`

## API Usage

### Health check

```bash
curl http://127.0.0.1:5000/health
```

Response:

```json
{ "status": "healthy" }
```

### Model info

```bash
curl http://127.0.0.1:5000/info
```

Response:

```json
{
  "model_type": "LogisticRegression",
  "features": [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
  ]
}
```

### Prediction example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Response:

```json
{
  "heart_disease": false,
  "probability": 0.1234
}
```

## Docker

Build the Docker image:

```bash
docker build -t heart-disease-api .
```

Run the container:

```bash
docker run --rm -p 5000:5000 heart-disease-api
```

## Docker Compose

This repository includes `docker-compose.yml` with the application and Redis service.

Start services:

```bash
docker-compose up --build
```

The FastAPI app will be available on `http://127.0.0.1:5001`.

## Notes

- `model/heart_disease_model.joblib` must exist before starting the app.
- Input feature order and names must match the trained model exactly.
- The model predicts a binary outcome and returns the probability for the positive class.
