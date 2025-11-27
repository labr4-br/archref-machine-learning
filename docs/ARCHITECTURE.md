# Architecture

## Containers Overview

```mermaid
graph TB
    CLIENT[HTTP Client] -->|POST /predict| API
    DEV[Developer] -->|access dashboard<br/>localhost:5000| MLF

    subgraph docker-compose
        direction LR
        API[api<br/>FastAPI :8000]
        TRN[train<br/>Pipeline]
        MLF[mlflow<br/>Server :5000]

        TRN -->|log experiments| MLF
        TRN -->|save model| VOL
        API -->|load model| VOL
        MLF -->|persist| MDATA

        subgraph Volumes
            VOL[(./models<br/>./data)]
            MDATA[(mlflow-data)]
        end
    end
```

## Services

| Service | Port | Function | Profile |
|---------|------|----------|---------|
| api | 8000 | REST API for predictions | default |
| mlflow | 5000 | Experiment tracking and model registry | mlflow |
| train | - | Run training pipeline | training |
| test | - | Run tests | testing |
| dev | 8000 | API with hot-reload | dev |

## Commands

```bash
# Production - API only
docker compose up api

# Development - API with hot-reload
docker compose --profile dev up dev

# Training - run pipeline
docker compose --profile training up train

# MLflow - tracking server
docker compose --profile mlflow up mlflow

# Tests
docker compose --profile testing up test
```

## Data Flow

```mermaid
sequenceDiagram
    participant D as Developer
    participant T as train container
    participant M as mlflow container
    participant A as api container
    participant C as Client

    D->>T: make train
    T->>T: Load data
    T->>T: Train model
    T->>M: Log metrics and parameters
    T->>M: Register model

    D->>A: docker compose up api
    A->>A: Load model (./models)

    C->>A: POST /predict
    A->>A: Inference
    A->>C: Result
```

## Volumes

| Volume | Container | Mode | Description |
|--------|-----------|------|-------------|
| ./models | api, train | rw/ro | Trained models |
| ./data | train | ro | Input data |
| ./config.yaml | api, train | ro | Configuration |
| ./reports | train | rw | Logs and metrics |
| mlflow-data | mlflow | rw | MLflow database and artifacts |

## Independence

- **API**: Works standalone with local model
- **MLflow**: Optional, used only for tracking
- **Train**: Can run with or without MLflow (`integrations.mlflow.enabled`)

## Roadmap

### DVC (Data Version Control)

Git-like data versioning.

```mermaid
graph LR
    DATA[data/raw] -->|dvc add| DVC[(DVC Cache)]
    DVC -->|dvc push| S3[(S3/GCS)]
    GIT[Git] -->|version| META[data/raw.dvc]
```

**Benefits:**
- Experiment reproducibility
- Dataset rollback
- Team sharing

### Pandera

Schema validation for DataFrames in the pipeline.

```mermaid
graph LR
    LOAD[Data Loader] -->|DataFrame| VAL{Pandera}
    VAL -->|valid| TRAIN[Train]
    VAL -->|invalid| ERR[DataValidationError]
```

**Benefits:**
- Detect corrupted data before training
- Document expected schema
- Automated quality tests

### MLflow Model Registry

Full model lifecycle with stages.

```mermaid
graph LR
    TRN[Train] -->|register| REG[(Registry)]
    REG --> STG[Staging]
    STG -->|approved| PRD[Production]
    PRD -->|rollback| STG
    PRD -->|deprecated| ARC[Archived]
```

**Benefits:**
- Deploy without API rebuild
- Instant rollback
- Approval before production
- Version audit

**Hybrid approach (production):**

MLflow manages **metadata and versioning**, but the model is loaded from the **local volume** for better performance.

```mermaid
graph LR
    TRN[Train] -->|register metadata| REG[(MLflow Registry)]
    TRN -->|save file| VOL[(Volume ./models)]
    REG -->|which version?| API
    VOL -->|load model| API
    API -->|predict| CLIENT
```

**Why hybrid?**
- No download latency at startup
- API works even with MLflow offline
- No cloud egress cost
- Scales horizontally without overloading MLflow

**Implementation:**
```python
import mlflow

# Query MLflow to know which version to use
client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions("RandomForest", stages=["Production"])[0]

# Load from local volume (path stored in metadata)
model = joblib.load(f"./models/{model_version.run_id}/model.pkl")
```

### Future Architecture

**Full view:**

```mermaid
graph TB
    subgraph Data Layer
        SRC[(Source)] -->|extract| DVC
        DVC -->|validate| PAN[Pandera]
    end

    subgraph ML Layer
        PAN -->|clean data| TRN[Train Pipeline]
        TRN -->|log metrics| MLF[MLflow Tracking]
        TRN -->|register metadata| REG[(Model Registry)]
        TRN -->|save file| VOL[(Volume)]
    end

    subgraph Registry
        REG --> STG[Staging]
        STG -->|promote| PRD[Production]
    end

    subgraph Serving Layer
        PRD -->|which version?| API[FastAPI]
        VOL -->|load model| API
        API -->|predict| CLIENT[Client]
    end
```

**Simplified flow:**

```mermaid
graph LR
    subgraph Data
        SRC[(Source)] --> DVC
        DVC --> PAN[Pandera]
    end

    subgraph Train
        PAN --> TRN[Pipeline]
        TRN --> MLF[MLflow]
        TRN --> VOL[(Volume)]
    end

    subgraph Registry
        MLF --> STG[Staging]
        STG -->|promote| PRD[Production]
    end
```

```mermaid
graph LR
    subgraph Serving
        direction TB

        subgraph Staging
            STG_R[Registry: Staging] --> API_S[staging.domain.com]
            VOL_S[(Volume)] --> API_S
            QA[QA] -->|validate| API_S
        end

        subgraph Production
            PRD_R[Registry: Production] --> API_P[api.domain.com]
            VOL_P[(Volume)] --> API_P
            CLI[Clients] -->|predict| API_P
        end
    end
```

**Environments by subdomain:**

| URL | MLflow Stage | Usage |
|-----|--------------|-------|
| `api.domain.com` | Production | Real clients |
| `staging.domain.com` | Staging | QA before promoting |
