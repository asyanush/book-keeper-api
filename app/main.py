"""Book Keeper – FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import cast

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ExceptionHandler

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.exceptions import (
    DomainException,
    domain_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.db.database import get_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_db().init_db()
    yield


app = FastAPI(title=settings.API_NAME, lifespan=lifespan)

# Register handlers centrally so every route returns the same error shape.
app.add_exception_handler(
    DomainException, cast(ExceptionHandler, domain_exception_handler)
)
app.add_exception_handler(
    RequestValidationError, cast(ExceptionHandler, validation_exception_handler)
)
app.add_exception_handler(
    StarletteHTTPException, cast(ExceptionHandler, http_exception_handler)
)
app.include_router(api_router, prefix="/v1")

@app.get("/")
def root():
    return {
        "name": settings.API_NAME,
        "version": settings.API_VERSION,
        "tagline": settings.APP_TAGLINE,
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {
        "name": settings.API_NAME,
        "version": settings.API_VERSION,
        "status": "healthy",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)
