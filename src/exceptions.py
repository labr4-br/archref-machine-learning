"""Custom exception hierarchy for ML ArchRef."""

from typing import Any


class ArchRefError(Exception):
    """Base exception for all ArchRef errors.

    All custom exceptions in this project should inherit from this class.
    """

    pass


# =============================================================================
# Data Exceptions
# =============================================================================


class DataError(ArchRefError):
    """Base exception for data-related errors."""

    pass


class DataLoadError(DataError):
    """Failed to load data from source.

    Raised when data cannot be loaded from file, database, or API.
    """

    def __init__(self, source: str, reason: str | None = None) -> None:
        self.source = source
        self.reason = reason
        message = f"Failed to load data from '{source}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class DataValidationError(DataError):
    """Data failed validation checks.

    Raised when data does not meet expected schema or quality requirements.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = f"Data validation failed with {len(errors)} error(s):\n"
        message += "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


class DataSchemaError(DataError):
    """Data schema mismatch.

    Raised when data columns or types don't match expected schema.
    """

    def __init__(
        self,
        expected: list[str],
        actual: list[str],
    ) -> None:
        self.expected = expected
        self.actual = actual
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        message = "Data schema mismatch:"
        if missing:
            message += f"\n  Missing columns: {missing}"
        if extra:
            message += f"\n  Unexpected columns: {extra}"
        super().__init__(message)


# =============================================================================
# Model Exceptions
# =============================================================================


class ModelError(ArchRefError):
    """Base exception for model-related errors."""

    pass


class ModelNotTrainedError(ModelError):
    """Operation requires a trained model.

    Raised when predict() or save() is called on an untrained model.
    """

    def __init__(self, operation: str = "operation") -> None:
        self.operation = operation
        super().__init__(
            f"Cannot perform '{operation}' on an untrained model. Call train() first."
        )


class ModelNotFoundError(ModelError):
    """Model file not found at specified path.

    Raised when attempting to load a model from a non-existent file.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Model file not found: {path}")


class ModelLoadError(ModelError):
    """Failed to load model from disk.

    Raised when model file exists but cannot be deserialized.
    """

    def __init__(self, path: str, reason: str | None = None) -> None:
        self.path = path
        self.reason = reason
        message = f"Failed to load model from '{path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ModelTrainingError(ModelError):
    """Error during model training.

    Raised when training fails due to data issues or algorithm errors.
    """

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Training failed for '{model_name}': {reason}")


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigError(ArchRefError):
    """Base exception for configuration errors."""

    pass


class ConfigNotFoundError(ConfigError):
    """Configuration file not found.

    Raised when the specified config file does not exist.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Configuration file not found: {path}")


class ConfigValidationError(ConfigError):
    """Configuration validation failed.

    Raised when config values fail Pydantic validation.
    """

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        message = f"Configuration validation failed with {len(errors)} error(s):\n"
        for err in errors:
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            message += f"  - {loc}: {msg}\n"
        super().__init__(message)


# =============================================================================
# Pipeline Exceptions
# =============================================================================


class PipelineError(ArchRefError):
    """Base exception for pipeline errors."""

    pass


class PipelineStepError(PipelineError):
    """A pipeline step failed.

    Raised when a specific step in the pipeline encounters an error.
    """

    def __init__(self, step: str, reason: str) -> None:
        self.step = step
        self.reason = reason
        super().__init__(f"Pipeline step '{step}' failed: {reason}")


class PipelineConfigError(PipelineError):
    """Pipeline configuration is invalid.

    Raised when pipeline cannot be constructed due to config issues.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Pipeline configuration error: {reason}")


# =============================================================================
# API Exceptions
# =============================================================================


class APIError(ArchRefError):
    """Base exception for API-related errors."""

    pass


class ModelNotLoadedError(APIError):
    """Model not loaded in API.

    Raised when prediction is requested but model is not available.
    """

    def __init__(self) -> None:
        super().__init__(
            "Model not loaded. Ensure the model file exists and API started correctly."
        )


class InvalidInputError(APIError):
    """Invalid input data for prediction.

    Raised when request data doesn't match expected format.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid input data: {reason}")
