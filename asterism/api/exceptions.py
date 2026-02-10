"""API exceptions and error handlers."""

from fastapi import Request
from fastapi.responses import JSONResponse

from asterism.llm.exceptions import AllProvidersFailedError


class APIError(Exception):
    """Base class for API errors."""

    def __init__(self, message: str, status_code: int = 500, code: str = "internal_error"):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401, code="invalid_api_key")


class ValidationError(APIError):
    """Raised when request validation fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400, code="invalid_request")


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.code,
                "code": exc.code,
            }
        },
    )


async def all_providers_failed_handler(request: Request, exc: AllProvidersFailedError) -> JSONResponse:
    """Handle AllProvidersFailedError exceptions."""
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": str(exc),
                "type": "all_providers_failed",
                "code": "service_unavailable",
            }
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "internal_error",
                "code": "internal_error",
            }
        },
    )
