from fastapi import APIRouter
from app.auth.router import router as auth_router
from app.books.router import router as books_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(books_router)