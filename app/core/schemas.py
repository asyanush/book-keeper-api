from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str
    
class ErrorResponse(BaseModel):
    """Error response schema."""

    message: str