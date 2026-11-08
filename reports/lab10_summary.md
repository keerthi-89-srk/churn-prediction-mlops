# Lab 10 - API serving and testing

## Model bundle

The API bundle was prepared from MLflow model `TelcoChurnClassifier`, Production
version 1, MLflow run `3482c67b7c5f47aeb1a2cd20ea674627`. The preprocessor was
downloaded from that same run's `preprocessing_pipeline/preprocessor.pkl`
artifact. Preparation checks the model against the preprocessor using a sample
from the project's raw dataset and records model provenance, metrics, and
SHA-256 checksums in `deployment/model_metadata.json`.

The local model registry and source model artifacts were not modified and the
model was not retrained. The bundle directory is ignored by Git because it
contains generated binary artifacts.

## API

The FastAPI application provides:

- `GET /` - basic service status.
- `GET /health` - healthy only when both model and preprocessor are loaded;
  otherwise returns HTTP 503.
- `GET /metadata` - model name, version, run ID, and readiness information.
- `POST /predict` - validates one customer and returns a churn prediction and
  probability.
- `POST /predict/batch` - validates and predicts between 1 and 20 customers.

Input validation rejects missing or extra fields, invalid categories,
non-finite or negative numeric values, malformed numeric charges, and
inconsistent blank `TotalCharges` (blank charges are accepted for tenure 0,
as in the source dataset). Inference logs contain the number of input fields
and prediction metadata, not customer input values or IDs.

## Tests

Executed with Python 3.10.10 in the existing virtual environment:

```text
python -m pytest tests\test_api.py tests\test_performance.py -q
36 passed, 1 warning in 24.86s
```

The warning is from Starlette's use of the deprecated AnyIO
`BlockingPortal` alias. The API tests cover the routes, valid and invalid
inputs, batch limits, readiness, and omission of customer values from logs.
Performance tests cover repeated success, consistency, batch response, and
basic latency/throughput measurements; no service-level latency target was
specified.

Install the project's API and test requirements into the virtual environment
with:

```powershell
python -m pip install -r requirements-api-test.txt
```
