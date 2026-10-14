# Iris ML Deployment Project

[![Train and test ML API](https://github.com/YOUR_USERNAME/ml-iris-deployment/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/ml-iris-deployment/actions/workflows/test.yml)



An end-to-end machine-learning example that trains a classifier, evaluates it on a held-out test set, serializes the full preprocessing/model pipeline with Joblib, and serves predictions through FastAPI.

## Features
- Built-in scikit-learn Iris dataset (no separate data download required)
- Stratified train/test split
- `StandardScaler` + Logistic Regression in one reproducible pipeline
- Evaluation metrics and confusion matrix saved as JSON
- Joblib model serialization
- FastAPI `/predict`, `/health`, and interactive `/docs` endpoints
- Input validation and automated API tests

## Project structure
```text
ml_iris_deployment/
├── app/main.py
├── models/                 # generated after training; not committed
├── scripts/train_model.py
├── tests/test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 1. Set up locally
Python 3.10+ is recommended.
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Train and evaluate
```bash
python scripts/train_model.py
```
This writes `models/iris_pipeline.joblib` and `models/metrics.json`. The script prints held-out test accuracy. Exact metrics may vary slightly by library version.

## 3. Start the API
```bash
uvicorn app.main:app --reload
```
- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

## 4. Make a prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
The response includes a predicted class and class probabilities.

## 5. Run tests
Train the model first, then:
```bash
pytest -q
```

## 6. Docker deployment
```bash
docker build -t iris-ml-api .
docker run --rm -p 8000:8000 iris-ml-api
```

## Deploy to a cloud host
This repository is container-ready. Push it to GitHub, then create a web service on a Docker-compatible host (for example, Render, Railway, or a cloud container service) from the repository. Configure the service to build from `Dockerfile` and expose port `8000`. Confirm the provider's current deployment settings before publishing. The model is trained during image build, so no model binary needs to be committed.

## API contract
`POST /predict` request:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
All measurements are in centimetres and must be greater than zero and no more than 10.

## Model card / limitations
- **Task:** educational multiclass classification of Iris species.
- **Input:** four numeric flower measurements.
- **Output:** setosa, versicolor, or virginica, with model-estimated probabilities.
- **Evaluation:** held-out 20% stratified test split; see generated `models/metrics.json`.
- **Limitations:** the dataset is small and clean. Results do not establish performance on arbitrary real-world flowers. Probabilities are model estimates, not guarantees.
- **Security:** Joblib/pickle artifacts can execute code when loaded. Load only trusted model files. Add authentication, rate limiting, logging, and monitoring before exposing an API publicly.

## GitHub publishing
1. Create a new empty GitHub repository named `ml-iris-deployment`.
2. Extract this ZIP and open a terminal in the project folder.
3. Run:
```bash
git init
git add .
git commit -m "Build end-to-end Iris ML prediction API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ml-iris-deployment.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your GitHub username. Do not commit secrets or virtual environments.


## Test it on GitHub
A GitHub Actions workflow is included at `.github/workflows/test.yml`. After you push this repository to GitHub, it automatically:
1. Installs the Python dependencies.
2. Trains and evaluates the model.
3. Runs the API tests with pytest.

Open the repository's **Actions** tab to view the latest run and logs. Replace `YOUR_USERNAME` in the README badge URL with your GitHub username after publishing.
