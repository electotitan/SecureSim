"""Uniform, traceback-free error responses.

Every error body has the shape::

    {"error": {"type": <ERROR_TYPE>, "message": <str>, "details": [ {...}, ... ]}}

``type`` lets the frontend distinguish the categories required by the specification:

    validation_error      malformed or out-of-range input (HTTP 422, 413, 4xx)
    transmission_failure  channel/correction could not deliver the data
    integrity_failure     checksum / packet / padding verification failed
    unsupported_operation request is well-formed but not supported (HTTP 501)
    internal_error        unexpected server fault (HTTP 500); no internal details are exposed
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("simulator")

ERROR_TYPES = (
    "validation_error", "transmission_failure", "integrity_failure",
    "unsupported_operation", "internal_error", "http_error",
)


def error_body(error_type: str, message: str, details: Optional[list[dict[str, Any]]] = None) -> dict:
    return {"error": {"type": error_type, "message": message, "details": details or []}}


class ValidationFailure(Exception):
    """Domain-level input problem (HTTP 422). The message is safe to show to the user."""

    def __init__(self, message: str, details: Optional[list[dict[str, Any]]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or []


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Only loc/msg/type are exposed: pydantic's 'input' and 'ctx' may echo user data or
        # contain non-JSON-serialisable exception objects.
        details = [
            {"field": ".".join(str(p) for p in err.get("loc", ())), "message": err.get("msg", ""),
             "code": err.get("type", "")}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422,
                            content=error_body("validation_error", "Request validation failed.", details))

    @app.exception_handler(ValidationFailure)
    async def _validation_failure(_: Request, exc: ValidationFailure) -> JSONResponse:
        return JSONResponse(status_code=422,
                            content=error_body("validation_error", exc.message, exc.details))

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 501:
            etype = "unsupported_operation"
        elif exc.status_code >= 500:
            etype = "internal_error"
        else:
            etype = "http_error"
        message = exc.detail if isinstance(exc.detail, str) else "Request could not be processed."
        return JSONResponse(status_code=exc.status_code, content=error_body(etype, message),
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")        # traceback goes to the server log only
        return JSONResponse(status_code=500,
                            content=error_body("internal_error", "Internal server error."))
