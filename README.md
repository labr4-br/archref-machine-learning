# ArchRef Machine Learning

A production-ready, modular Machine Learning project template with MLflow integration.

## Overview

This ArchRef provides a solid foundation for ML projects with:

- **Modular Architecture** - Core base classes + task-specific plugins (classification, regression, clustering)
- **MLflow Integration** - Experiment tracking, model registry, and artifact management
- **Configuration-Driven** - YAML + Pydantic validation with environment variable overrides
- **Production API** - FastAPI with task-specific endpoints
- **Quality First** - Type hints, tests, pre-commit hooks, and CI/CD

## Quick Start

### Prerequisites

- Python 3.10+
- pip or uv

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd archref-machine-learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install with all task plugins
pip install -e ".[all,dev]"

# Or install specific task only
pip install -e ".[classification,dev]"
```

### Training a Model

```bash
# Run the training pipeline
make train

# Or directly
python -m src.pipeline --config config.yaml
```

### Starting the API

```bash
# Start the API server
make serve

# Or directly
uvicorn src.app.main:app --reload --port 8000
```

### Making Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"f0": 0.5, "f1": 0.3, "f2": 0.8, "f3": 0.1, "f4": 0.9}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ARCHREF ML                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      CORE                                │   │
│  │  BaseModel  │  BaseDataLoader  │  BaseMetrics           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│            ┌───────────────┼───────────────┐                   │
│            ▼               ▼               ▼                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │Classification│ │  Regression  │ │  Clustering  │           │
│  │    Plugin    │ │    Plugin    │ │    Plugin    │           │
│  │              │ │              │ │              │           │
│  │ • metrics    │ │ • metrics    │ │ • metrics    │           │
│  │ • models     │ │ • models     │ │ • models     │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
│                            │                                    │
│            ┌───────────────┼───────────────┐                   │
│            ▼               ▼               ▼                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │   MLflow     │ │   FastAPI    │ │   Pipeline   │           │
│  │  Tracking    │ │   Serving    │ │ Orchestrator │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
archref-machine-learning/
├── config.yaml              # Project configuration
├── pyproject.toml           # Dependencies and tool configs
├── Makefile                 # Development commands
├── Dockerfile               # Container definition
├── docker-compose.yml       # Service orchestration
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Pydantic configuration models
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── pipeline.py          # Training pipeline orchestrator
│   │
│   ├── core/                # Base classes (always present)
│   │   ├── __init__.py
│   │   ├── base_model.py    # Abstract model interface
│   │   ├── base_loader.py   # Abstract data loader
│   │   └── base_metrics.py  # Abstract metrics calculator
│   │
│   ├── tasks/               # Task-specific plugins
│   │   ├── __init__.py
│   │   ├── classification/  # Classification task
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py   # Accuracy, F1, ROC, etc.
│   │   │   └── models.py    # RandomForest, XGBoost, etc.
│   │   ├── regression/      # Regression task (future)
│   │   └── clustering/      # Clustering task (future)
│   │
│   ├── integrations/        # External tool wrappers
│   │   ├── __init__.py
│   │   └── mlflow_tracker.py
│   │
│   ├── data/                # Data loading
│   │   ├── __init__.py
│   │   ├── base_loader.py
│   │   └── mock_loader.py
│   │
│   ├── app/                 # API serving
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── config_loader.py
│       └── logger.py
│
├── tests/                   # Test suite
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── api/
│
├── data/                    # Data directories
│   ├── raw/
│   ├── processed/
│   ├── interim/
│   └── external/
│
├── models/                  # Trained model artifacts
├── notebooks/               # Jupyter notebooks
└── reports/
    ├── figures/
    └── logs/
```

## Configuration

### config.yaml

```yaml
project:
  name: "My ML Project"
  version: "0.1.0"

task:
  type: "classification"  # classification | regression | clustering

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  models: "models"
  logs: "reports/logs/app.log"

model:
  name: "RandomForest"
  params:
    n_estimators: 100
    random_state: 42
    max_depth: 10

training:
  test_size: 0.2
  random_state: 42

integrations:
  mlflow:
    enabled: true
    tracking_uri: "http://localhost:5000"
    experiment_name: "my-experiment"
```

### Environment Variables

Override any config with `ML_` prefix:

```bash
export ML_PROJECT_NAME="Production Model"
export ML_INTEGRATIONS__MLFLOW__ENABLED=true
```

## Usage Examples

### Classification Task

```python
from src.core.base_model import BaseModel
from src.tasks.classification.metrics import ClassificationMetrics
from src.tasks.classification.models import RandomForestModel
from src.integrations.mlflow_tracker import MLflowTracker
from src.config import load_config

# Load configuration
config = load_config("config.yaml")

# Initialize model and tracker
model = RandomForestModel(config.model.params)
tracker = MLflowTracker(config.integrations.mlflow)
metrics_calculator = ClassificationMetrics()

# Train with tracking
with tracker.start_run(run_name="experiment-1"):
    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = metrics_calculator.calculate(y_test, predictions)

    tracker.log_params(config.model.params)
    tracker.log_metrics(metrics)
    tracker.log_model(model, "model")
```

### Adding a Custom Model

```python
# src/tasks/classification/models.py
from src.core.base_model import BaseModel

class MyCustomModel(BaseModel):
    task_type = "classification"

    def __init__(self, params: dict):
        super().__init__(params)
        # Initialize your model

    def train(self, X, y) -> None:
        # Training logic
        pass

    def predict(self, X):
        # Prediction logic
        pass
```

## Development

### Setup

```bash
make dev  # Install dev dependencies + pre-commit hooks
```

### Commands

```bash
make help      # Show all commands
make train     # Run training pipeline
make serve     # Start API server
make test      # Run tests with coverage
make lint      # Run linters (ruff + mypy)
make format    # Format code
make docker-up # Start services with Docker
```

### Code Quality

```bash
# Pre-commit runs automatically on commit
# Manual run:
pre-commit run --all-files
```

## API Reference

### Endpoints by Task Type

| Task | Method | Endpoint | Description |
|------|--------|----------|-------------|
| All | GET | `/` | Health check |
| All | GET | `/health` | Detailed health status |
| All | GET | `/model/info` | Model metadata |
| Classification | POST | `/predict` | Predict class |
| Regression | POST | `/predict` | Predict value |
| Clustering | POST | `/cluster` | Assign cluster |
| Anomaly Detection | POST | `/detect` | Detect anomaly |

### Classification: `/predict`

**Request:**
```json
{
  "f0": 0.5,
  "f1": 0.3,
  "f2": 0.8,
  "f3": 0.1,
  "f4": 0.9
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": [0.3, 0.7],
  "model_version": "1.0.0"
}
```

### Clustering: `/cluster`

**Request:**
```json
{
  "f0": 0.5,
  "f1": 0.3,
  "f2": 0.8,
  "f3": 0.1,
  "f4": 0.9
}
```

**Response:**
```json
{
  "cluster_id": 2,
  "distance_to_centroid": 0.45,
  "model_version": "1.0.0"
}
```

### Anomaly Detection: `/detect`

**Request:**
```json
{
  "f0": 0.5,
  "f1": 0.3,
  "f2": 0.8,
  "f3": 0.1,
  "f4": 0.9
}
```

**Response:**
```json
{
  "is_anomaly": true,
  "anomaly_score": 0.87,
  "model_version": "1.0.0"
}
```

## Deployment

### Docker

```bash
# Build and start
docker-compose up -d api

# Run training in container
docker-compose run --rm train

# View logs
docker-compose logs -f api
```

### Production Checklist

- [ ] Set `ML_` environment variables
- [ ] Configure MLflow tracking server
- [ ] Mount volumes for models and data
- [ ] Set resource limits
- [ ] Configure health checks
- [ ] Enable HTTPS

## Testing

```bash
# Run all tests
make test

# Run with coverage report
pytest -v --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py -v
```

## Roadmap

### Task Plugins
- [ ] Regression Plugin (MSE, RMSE, MAE, R² + Linear, Ridge, XGBoost)
- [ ] Clustering Plugin (Silhouette, Calinski-Harabasz + KMeans, DBSCAN)
- [ ] Time Series Plugin (MAPE, forecast metrics + ARIMA, Prophet)
- [ ] Anomaly Detection Plugin (Isolation Forest, One-Class SVM)

### Architecture
- [ ] Plugin Registry Pattern with auto-discovery
- [ ] Dependency Injection Container
- [ ] Pipeline DAG (Kedro-style)

### Integrations
- [ ] DVC for data versioning
- [ ] Feature Store (Feast)
- [ ] Great Expectations for data validation
- [ ] Evidently AI for model monitoring

### Serving
- [ ] Batch prediction endpoint
- [ ] Model versioning in API
- [ ] A/B Testing support
- [ ] BentoML/Seldon integration

### Infrastructure
- [ ] Kubernetes Helm charts
- [ ] Airflow/Prefect DAGs
- [ ] Terraform modules

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat(scope): add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Convention

```
feat(scope): add new feature
fix(scope): fix bug
docs(scope): update documentation
test(scope): add tests
refactor(scope): refactor code
chore(scope): maintenance tasks
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
